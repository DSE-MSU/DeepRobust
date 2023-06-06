import numpy as np
import os
import json
import torch
from utils import *


class GPT2_detector:
    def __init__(self, pretrained_model=True, n_samples=500):
        self.pretrained_model = pretrained_model
        self.n_samples = n_samples

    def main_func(self, data, model_collector, results_path, batch_size):
        if self.pretrained_model:
            output = run_detect_experiment(data, model_collector, batch_size, self.n_samples)
            with open(os.path.join(results_path, 'results.json'), 'w') as f:
                json.dump(output, f)

def run_detect_experiment(data, model_collector, batch_size, n_samples=500):
    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    # get predictions for original text
    with torch.no_grad():
        real_preds = []
        for batch in range(len(original_text) // batch_size):
            batch_real = original_text[batch * batch_size:(batch + 1) * batch_size]
            tokenized_batch_real = model_collector.classification_tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_collector.classification_model.device)
            pred_probs = model_collector.classification_model(input_ids=tokenized_batch_real['input_ids'], attention_mask=tokenized_batch_real['attention_mask'])
            pred_probs = torch.softmax(pred_probs[0], dim=-1)
            #fake, real = probs.detach().cpu().flatten().numpy().tolist()
            real_preds.extend(pred_probs[:,1].cpu().numpy().tolist())
        
        # get predictions for sampled text
        fake_preds = []
        for batch in range(len(sampled_text) // batch_size):
            batch_fake = sampled_text[batch * batch_size:(batch + 1) * batch_size]
            tokenized_batch_fake = model_collector.classification_tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_collector.classification_model.device)
            pred_probs = model_collector.classification_model(input_ids=tokenized_batch_fake['input_ids'], attention_mask=tokenized_batch_fake['attention_mask'])
            pred_probs = torch.softmax(pred_probs[0], dim=-1)

            fake_preds.extend(pred_probs[:,0].cpu().numpy().tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    return {
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
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
    

def batch_predict(model, tokenized_texts):
    with torch.no_grad():
        try:
            pred_probs = model(input_ids=tokenized_texts['input_ids'], attention_mask=tokenized_texts['attention_mask'])
            pred_probs = torch.softmax(pred_probs[0], dim=-1)
            return pred_probs
        except:
            pred_probs = []
            max_length = tokenized_texts['input_ids'].size()[-1]

            for i in range(len(tokenized_texts['input_ids'])):
                pred_prob = model(input_ids=tokenized_texts['input_ids'][i].reshape(1, max_length), attention_mask=tokenized_texts['attention_mask'][i].reshape(1, max_length))
                pred_prob = torch.softmax(pred_prob[0], dim=-1)
                pred_probs.append(pred_prob)

            return torch.squeeze(torch.stack(pred_probs), 1)


