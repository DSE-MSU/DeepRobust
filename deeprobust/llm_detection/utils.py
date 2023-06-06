import numpy as np
import random
import os
import re
import subprocess
import torch
import transformers
from datasets import load_dataset
import functools
from multiprocessing.pool import ThreadPool
from sklearn.metrics import roc_curve, precision_recall_curve, auc


class model_collector:
    def __init__(self, base_model=None, base_tokenizer=None, mask_model=None, mask_tokenizer=None, classification_model=None, classification_tokenizer=None, device='cuda', openai_model=None, 
                 do_top_k=False, top_k=40, do_top_p=False, top_p=0.96):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.mask_model = mask_model
        self.mask_tokenizer = mask_tokenizer
        self.classification_model = classification_model
        self.classification_tokenizer = classification_tokenizer
        self.device = device
        self.openai_model = openai_model
        self.do_top_k = do_top_k
        self.top_k = top_k
        self.do_top_p = do_top_p
        self.top_p = top_p


    def load_base_model(self):
        try:
            self.mask_model.cpu()
        except:
            pass
        if self.openai_model is None:
            self.base_model.to(self.device)
        print(f'BASE model has moved to GPU', flush=True)


    def load_mask_model(self):
        if self.openai_model is None:
            self.base_model.cpu()

        self.mask_model.to(self.device)
        print(f'MASK model has moved to GPU', flush=True) 


    def load_classfication_model(self):
        if self.openai_model is None:
            self.base_model.cpu()

        self.classification_model.to(self.device)
        print(f'CLASSIFICATION model has moved to GPU', flush=True) 
        

class data_preprocess:
    def __init__(self, dataset_name, cache_dir, batch_size=50, min_words=55, prompt_tokens=30):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.min_words = 30 if dataset_name in ['pubmed'] else min_words
        self.prompt_tokens = prompt_tokens


    def generate_data_predefined(self):
        # DATASET = ['xsum', 'squad', 'writing', 'pubmed', 'wmt16_en', 'wmt16_de']
        if self.dataset_name == 'xsum':
            self.data = load_dataset('xsum', split='train', cache_dir=self.cache_dir)['document']
        elif self.dataset_name == 'squad':
            self.data = load_dataset('squad', split='train', cache_dir=self.cache_dir)['context']
        elif self.dataset_name == 'pubmed':
            self.data = load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=self.cache_dir)
            # combine question and long_answer
            self.data = [f'Question: {q} Answer:<<<SEP>>>{a}' for q, a in zip(self.data['question'], self.data['long_answer'])]
        elif self.dataset_name == 'wmt16_en':
            self.data = load_language('en', self.cache_dir)
        elif self.dataset_name == 'wmt16_de':
            self.data = load_language('de', self.cache_dir)
        elif self.dataset_name == 'writing':
            writing_path = cache_dir + '/writingPrompts'
            if os.path.isdir(writing_path):
                with open(f'{writing_path}/valid.wp_source', 'r') as f:
                    prompts = f.readlines()
                with open(f'{writing_path}/valid.wp_target', 'r') as f:
                    stories = f.readlines()
                
                prompts = [self.process_prompt(prompt) for prompt in prompts]
                joined = [self.process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
                self.data = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]
            else:
                raise ValueError(f'Dataset WritingPrompts is not existed. Please download it first and unzip it into ./data folder')
        else:
            raise ValueError(f'Dataset {self.dataset_name} is not included.')

        # get unique examples, strip whitespace, and remove newlines
        # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
        # then take just the examples that are <= 512 tokens (for the mask model)
        # then generate n_samples samples

        # remove duplicates from the data
        self.data = list(dict.fromkeys(self.data))  # deterministic, as opposed to set()

        # strip whitespace around each example
        self.data = [x.strip() for x in self.data]

        # remove newlines from each example
        self.data = [' '.join(x.split()) for x in self.data]

        # try to keep only examples with > 250 words
        if self.dataset_name in ['xsum', 'squad', 'writing']:
            long_data = [x for x in self.data if len(x.split()) > 250]
            if len(long_data) > 0:
                self.data = long_data

        random.shuffle(self.data)


    def generate_data_customized(self, key_name, load_hf_dataset=True):
        if load_hf_dataset:
            self.data = load_dataset(dataset_name, split='train', cache_dir=self.cache_dir)[key_name]
        else:
            with open(f'./data/{dataset_name}', 'r') as f:
                self.data = f.readlines()

        random.shuffle(self.data)


def load_base_model_and_tokenizer(base_model_name, openai_model, dataset_name, cache_dir):
    if openai_model is None:
        print(f'Loading BASE model {base_model_name}...')
        base_model_kwargs = {}
        if 'gpt-j' in base_model_name or 'neox' in base_model_name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in base_model_name:
            base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in base_model_name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if dataset_name in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def generate_samples(raw_data, dataset, model_collector, dataset_name):
    data = {
        "original": [],
        "sampled": [],
    }

    for batch in range(len(raw_data) // dataset.batch_size):
        print(f'Generating samples for batch {batch} of {len(raw_data) // dataset.batch_size}', flush=True)
        original_text = raw_data[batch * dataset.batch_size:(batch + 1) * dataset.batch_size]
        sampled_text = sample_from_model(dataset, original_text, model_collector, '<<<SEP>>>')

        for o, s in zip(original_text, sampled_text):
            if dataset_name == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace('<<<SEP>>>', ' ')

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)
    
    # if pre_perturb_pct > 0:
    #     print(f'APPLYING {pre_perturb_pct}, {pre_perturb_span_length} PRE-PERTURBATIONS', flush=True)
    #     load_mask_model()
    #     data["sampled"] = perturb_texts(data["sampled"], model_collector, random_fills, random_fills_tokens, pre_perturb_span_length, pre_perturb_pct, chunk_size, ceil_pct=True)
    #     load_base_model()

    return data


def load_language(language, cache_dir):
        # load either the english or german portion of the wmt16 dataset
        assert language in ['en', 'de']
        d = load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
        docs = d['translation']
        desired_language_docs = [d[language] for d in docs]
        lens = [len(d.split()) for d in desired_language_docs]
        sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
        return sub


def process_spaces(story):
        return story.replace(
            ' ,', ',').replace(
            ' .', '.').replace(
            ' ?', '?').replace(
            ' !', '!').replace(
            ' ;', ';').replace(
            ' \'', '\'').replace(
            ' ’ ', '\'').replace(
            ' :', ':').replace(
            '<newline>', '\n').replace(
            '`` ', '"').replace(
            ' \'\'', '"').replace(
            '\'\'', '"').replace(
            '.. ', '... ').replace(
            ' )', ')').replace(
            '( ', '(').replace(
            ' n\'t', 'n\'t').replace(
            ' i ', ' I ').replace(
            ' i\'', ' I\'').replace(
            '\\\'', '\'').replace(
            '\n ', '\n').strip()


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(dataset, texts, model_collector, separator):
    # encode each text as a list of token ids
    if dataset.dataset_name == 'pubmed':
        texts = [t[:t.index(separator)] for t in texts]
        all_encoded = model_collector.base_tokenizer(texts, return_tensors="pt", padding=True).to(model_collector.base_model.device)
    else:
        all_encoded = model_collector.base_tokenizer(texts, return_tensors="pt", padding=True).to(model_collector.base_model.device)
        all_encoded = {key: value[:, :dataset.prompt_tokens] for key, value in all_encoded.items()}

    if model_collector.openai_model:
        # decode the prefixes back into text
        prefixes = model_collector.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(dataset.batch_size)

        #decoded = pool.map(_openai_sample, dataset, model_collector, prefixes)
        func = functools.partial(_openai_sample, dataset=dataset, model_collector=model_collector)
        decoded = pool.map(func, prefixes)
    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < dataset.min_words:
            if tries != 0:
                print(f'min words: {m}, needed {dataset.min_words}, regenerating (try {tries})', flush=True)

            sampling_kwargs = {}
            if model_collector.do_top_p:
                sampling_kwargs['top_p'] = model_collector.top_p
            elif model_collector.do_top_k:
                sampling_kwargs['top_k'] = model_collector.top_k
            min_length = 50 if dataset.dataset_name in ['pubmed'] else 150
            outputs = model_collector.base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=model_collector.base_tokenizer.eos_token_id, eos_token_id=model_collector.base_tokenizer.eos_token_id)
            decoded = model_collector.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    return decoded


def _openai_sample(dataset, model_collector, p):
    if dataset.dataset_name != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": model_collector.openai_model, "max_tokens": 200 }
    if dataset.do_top_p:
        kwargs['top_p'] = dataset.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def tokenize_and_mask(text, span_length, pct, ceil_pct=False, buffer_size=1):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(model_collector, texts, mask_top_p=1.0):
    n_expected = count_masks(texts)
    stop_id = model_collector.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = model_collector.mask_tokenizer(texts, return_tensors="pt", padding=True).to(model_collector.mask_model.device)
    outputs = model_collector.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return model_collector.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # define regex to match all <extra_id_*> tokens, where * is an integer
    pattern = re.compile(r"<extra_id_\d+>")

    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


# Get the log likelihood of each text under the base_model
def get_ll(text, model_collector):
    if model_collector.openai_model:    
        import openai    
        kwargs = { "engine": model_collector.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = model_collector.base_tokenizer(text, return_tensors="pt").to(model_collector.base_model.device)
            labels = tokenized.input_ids
            return -model_collector.base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts, model_collector, batch_size):
    if not model_collector.openai_model:
        return [get_ll(text, model_collector) for text in texts]
    else:
        # global API_TOKEN_COUNTER

        # # use GPT2_TOKENIZER to get total number of tokens
        # total_tokens = sum(len(GPT2_TOKENIZER.encode(text)) for text in texts)
        # API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(batch_size)
        func = functools.partial(get_ll, model_collector=model_collector)
        return pool.map(func, texts)
        #return pool.map(get_ll, texts)


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

