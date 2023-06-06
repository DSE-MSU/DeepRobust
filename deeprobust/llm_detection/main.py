import argparse

parser = argparse.ArgumentParser(description='text test')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--method', default='gpt2_detector', type=str, help='name of detection method')
parser.add_argument('--target_text', default='', type=str, help='text for detection')
parser.add_argument('--customized_dataset', type=str, default='', help='whether using customized dataset for detection')
parser.add_argument('--dataset', default='xsum', type=str, help='dataset used for model training and evaluation')
parser.add_argument('--load_hf_dataset', type=str, default='', help='whether loading a dataset from huggingface')
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--device', default='cuda', type=str, help='which device used to execute')
parser.add_argument('--base_model_name', type=str, default='gpt2-medium', help='model to generate texts for detection')

parser.add_argument('--openai_model', type=str, default=None)
parser.add_argument('--openai_key', type=str, default='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')
parser.add_argument('--cache_dir', type=str, default='./cache_dir', help='path of the folder to save cache')
parser.add_argument('--output_to_log', default='', help='whether saving logs into a file')

parser.add_argument('--mask_filling_model_name', type=str, default='t5-large', help='name of mask model used in detect_gpt')
parser.add_argument('--pretrained_model', type=str, default='yes', help='whether using a pretrained detector in gpt2_detector')
parser.add_argument('--pretrained_model_name', type=str, default='roberta-base', help='name of model used in gpt2_detector')

parser.add_argument('--local', default='', type=str, help='the gpu number used on developing node.')

args = parser.parse_args()

import os
if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import numpy as np
import torch
import random
import sys
import json
import subprocess
import transformers
from utils import *
from detect_gpt import *
from gpt2_detector import *


def main():
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # cudnn.deterministic = True

    print(f'Seed number is {args.seed}', flush=True)

    device = args.device
    print('Use GPU to run', flush=True) if device == 'cuda' else print('Use CPU to run', flush=True)

    if args.target_text:
        pass
    else:
        print(f'Use DATASET {args.dataset} to generate texts for detection')

        if args.openai_model is None:
            base_model_name = args.base_model_name.replace('/', '_')
        else:
            base_model_name = "openai-" + args.openai_model.replace('/', '_')

        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)

        data_path = f'./detection_results/{args.dataset}/{base_model_name}'
        results_path = os.path.join(data_path, args.method)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        print(f'Saving results to absolute path: {os.path.abspath(results_path)}')

        if args.output_to_log:
            log = open(os.path.join(results_path, 'outputs.log'), 'w')
            sys.stdout = log
            print(f'Saving logs to absolute path: {os.path.abspath(log_path)}')

        # load dataset
        print(f'Loading DATASET {args.dataset}...', flush=True)
        if os.path.isfile(os.path.join(data_path, 'raw_data.json')):
            with open(os.path.join(data_path, 'raw_data.json')) as f:
                final_data = json.load(f)

            base_model = None
        else:
            cur_dataset = data_preprocess(args.dataset, args.cache_dir) 
            if args.customized_dataset:
                cur_dataset.generate_data_customized(args.key_name, args.load_hf_dataset)
            else:
                cur_dataset.generate_data_predefined()

            preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=args.cache_dir)
            if args.dataset in ['wmt16_en', 'wmt16_de']:
                mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_filling_model_name, model_max_length=512, cache_dir=args.cache_dir)
                preproc_tokenizer = mask_tokenizer

            # keep only examples with <= 512 tokens according to mask_tokenizer
            # this step has the extra effect of removing examples with low-quality/garbage content
            data = cur_dataset.data[:5000]
            tokenized_data = preproc_tokenizer(data)
            data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

            # print stats about remainining data
            print(f"Total number of samples: {len(data)}", flush=True)
            print(f"Average number of words: {np.mean([len(x.split()) for x in data])}", flush=True)

            # define generative model which generates texts
            base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name, args.openai_model, args.dataset, args.cache_dir)

            cur_model = model_collector(base_model=base_model, base_tokenizer=base_tokenizer, device=device, openai_model=args.openai_model)
            cur_model.load_base_model()

            final_data = generate_samples(data[:args.n_samples], cur_dataset, cur_model, args.dataset)

            # write the data to a json file in the save folder
            with open(os.path.join(data_path, 'raw_data.json'), 'w') as f:
                print(f"Writing raw data to {os.path.join(data_path, 'raw_data.json')}")
                json.dump(final_data, f)

        if base_model is None:
            base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name, args.openai_model, args.dataset, args.cache_dir)

        if args.openai_model is not None:
            import openai
            assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
            openai.api_key = args.openai_key
        else:
            base_model = base_model.to(device)


        if args.method == 'detect_gpt':
            if args.mask_filling_model_name:
                # mask filling t5 model
                print(f'Loading mask filling model {args.mask_filling_model_name}...', flush=True)
                mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model_name, cache_dir=args.cache_dir)
                try:
                    n_positions = mask_model.config.n_positions
                except AttributeError:
                    n_positions = 512
            else:
                mask_model = None
                n_positions = 512

            mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_filling_model_name, model_max_length=n_positions, cache_dir=args.cache_dir)

            cur_models = model_collector(base_model=base_model, base_tokenizer=base_tokenizer, mask_model=mask_model, mask_tokenizer=mask_tokenizer, device=device, openai_model=args.openai_model)
            cur_models.load_base_model()            

            detect_method = DetectGPT()
            detect_method.main_func(final_data, cur_models, results_path, args.mask_filling_model_name, args.batch_size)
        elif args.method == 'gpt2_detector':
            if args.pretrained_model == 'yes':
                pretrained_model = True

                import urllib.request
                from transformers import RobertaForSequenceClassification, RobertaTokenizer

                assert args.pretrained_model_name in ['roberta-base', 'roberta-large'], "Must provide an effective pretrained model name, either 'roberta-base' or 'roberta-large'"

                if args.pretrained_model_name == 'roberta-base':
                    download_path = 'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt'
                    saved_path = args.cache_dir + '/detector-base.pt'
                else:
                    download_path = 'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt'
                    saved_path = args.cache_dir + '/detector-large.pt'
                if not os.path.exists(saved_path):
                    urllib.request.urlretrieve(download_path, saved_path)

                print(f'Loading checkpoint from {saved_path}', flush=True)
                para = torch.load(saved_path, map_location='cpu')

                classification_model = RobertaForSequenceClassification.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)
                classification_tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)

                classification_model.load_state_dict(para['model_state_dict'], strict=False)
                classification_model.eval()
            else:
                pretrained_model = False

            cur_models = model_collector(base_model=base_model, base_tokenizer=base_tokenizer, classification_model=classification_model, classification_tokenizer=classification_tokenizer, device=device, openai_model=args.openai_model)
            cur_models.load_classfication_model()

            detect_method = GPT2_detector(pretrained_model)
            detect_method.main_func(final_data, cur_models, results_path, args.batch_size)



if __name__ == "__main__":
    main()  


