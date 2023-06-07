import numpy as np
import os
import json
from utils import *


class DetectGPT:
    def __init__(self, n_samples=500, n_perturbation_list='1,10,100', n_perturbation_rounds=1, span_length=2, pct_words_masked=0.3, pre_perturb_pct=0.0, pre_perturb_span_length=5, chunk_size=20, mask_top_p=1.0):
        self.n_samples = n_samples
        self.n_perturbation_list = [int(x) for x in n_perturbation_list.split(",")]
        self.n_perturbation_rounds = n_perturbation_rounds
        self.span_length = span_length
        self.pct_words_masked = pct_words_masked
        self.pre_perturb_pct = pre_perturb_pct
        self.pre_perturb_span_length = pre_perturb_span_length
        self.chunk_size = chunk_size
        self.mask_top_p = mask_top_p

    def main_func(self, data, model_collector, results_path, mask_filling_model_name, batch_size):
        # run perturbation experiments
        outputs = []
        for n_perturbations in self.n_perturbation_list:
            perturbation_results = get_perturbation_results(data, model_collector, self.pct_words_masked, self.mask_top_p, self.n_perturbation_rounds, self.span_length, n_perturbations, self.n_samples, self.chunk_size, mask_filling_model_name, batch_size=batch_size)
            for perturbation_mode in ['d', 'z']:
                output = run_perturbation_experiment(
                    perturbation_results, perturbation_mode, pct_words_masked=self.pct_words_masked, span_length=self.span_length, n_perturbations=n_perturbations, n_samples=self.n_samples)
                outputs.append(output)
                with open(os.path.join(results_path, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"), "w") as f:
                    json.dump(output, f)


def perturb_texts(texts, model_collector, mask_top_p, span_length, pct, chunk_size, ceil_pct=False, mask_filling_model_name='t5-large'):
    if '11b' in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in range(0, len(texts), chunk_size):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], model_collector, mask_top_p, span_length, pct, ceil_pct=ceil_pct))
    return outputs


def perturb_texts_(texts, model_collector, mask_top_p, span_length, pct, ceil_pct=False, buffer_size=1):
    #if not random_fills:
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct, buffer_size) for x in texts]
    raw_fills = replace_masks(model_collector, masked_texts, mask_top_p)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].', flush=True)
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct, buffer_size) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(model_collector, masked_texts, mask_top_p)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return perturbed_texts


def get_perturbation_results(data, model_collector, pct_words_masked, mask_top_p, n_perturbation_rounds, span_length=10, n_perturbations=1, n_samples=500, chunk_size=20, mask_filling_model_name='t5-large', batch_size=50):
    model_collector.load_mask_model()

    results = []
    original_text = data["original"][:n_samples]
    sampled_text = data["sampled"][:n_samples]

    perturb_fn = functools.partial(perturb_texts, model_collector=model_collector, mask_top_p=mask_top_p, span_length=span_length, pct=pct_words_masked, chunk_size=chunk_size, mask_filling_model_name=mask_filling_model_name)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
        except AssertionError:
            break

    assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    model_collector.load_base_model()

    for res in results:
        p_sampled_ll = get_lls(res["perturbed_sampled"], model_collector, batch_size)
        p_original_ll = get_lls(res["perturbed_original"], model_collector, batch_size)
        res["original_ll"] = get_ll(res["original"], model_collector)
        res["sampled_ll"] = get_ll(res["sampled"], model_collector)
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results


def run_perturbation_experiment(results, criterion, pct_words_masked, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print('WARNING: std of perturbed original is 0, setting to 1')
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print('WARNING: std of perturbed sampled is 0, setting to 1', flush=True)
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}", flush=True)
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }    
