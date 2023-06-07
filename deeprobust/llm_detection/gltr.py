import numpy as np
import os
import json
from utils import *


class GLTR:
    def __init__(self, n_samples=500):
        self.n_samples = n_samples

    def main_func(self, data, model_collector, results_path, batch_size):
        output = run_detect_experiment(data, model_collector, batch_size, self.n_samples)
        with open(os.path.join(results_path, 'results.json'), 'w') as f:
            json.dump(output, f)


def run_detect_experiment(data, model_collector, batch_size, n_samples=500):
    results = []

    original_text = data["original"][:n_samples]
    sampled_text = data["sampled"][:n_samples]

    results = []
    for batch in range(n_samples // batch_size):
        original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "original_crit": get_ll(original_text[idx], model_collector),
                "sampled": sampled_text[idx],
                "sampled_crit": get_ll(sampled_text[idx], model_collector),
            })

    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results],
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'predictions': predictions,
        'info': {
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
