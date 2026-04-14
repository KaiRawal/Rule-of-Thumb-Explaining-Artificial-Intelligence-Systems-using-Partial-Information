import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import shap
import pandas as pd
import os
import re
from lime.lime_text import LimeTextExplainer
from captum.attr import LayerIntegratedGradients, LLMGradientAttribution, TextTokenInput
import time
import argparse

# Load model and tokenizer once at import
tokenizer = AutoTokenizer.from_pretrained("L-NLProc/PredEx_RoBERTa_Large_Pred")
model = AutoModelForSequenceClassification.from_pretrained(
    "L-NLProc/PredEx_RoBERTa_Large_Pred", trust_remote_code=True
)
model.eval()




# ---------- SHAP Prediction Utilities ----------

def predict_confidence(input_passage: str) -> float:
    encoding = tokenizer(input_passage, return_tensors='pt', truncation=False, padding=False)
    input_ids = encoding['input_ids'][0]
    max_len, stride = 510, 410
    total_len = len(input_ids)
    confidence_scores = []

    for start in range(0, total_len, stride):
        end = min(start + max_len, total_len)
        chunk_ids = input_ids[start:end]
        chunk = tokenizer.build_inputs_with_special_tokens(chunk_ids.tolist())
        input_tensor = torch.tensor([chunk])
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            outputs = model(input_tensor, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()
            confidence_scores.append(confidence)
    # print(confidence_scores)
    return float(np.mean(confidence_scores)) if confidence_scores else None

def predict_proba_batch(input_texts: list[str]) -> np.ndarray:
    return np.array([[1 - (conf := predict_confidence(text)), conf] for text in input_texts])

# ---------- SHAP Explanation ----------

def shap_explain_text_to_csv(text: str, output_filename: str, nsamps=10, overwrite=False):
    print(f'SHAP: explaining text with {len(text)} characters')
    os.makedirs(f'DATA/SHAP_TOKENS_{nsamps}', exist_ok=True)
    output_filepath = f'DATA/SHAP_TOKENS_{nsamps}/{output_filename}.csv'
    if not overwrite and os.path.exists(output_filepath):
        return
    start_time = time.time()
    explainer = shap.Explainer(predict_proba_batch, tokenizer, max_evals=nsamps)
    shap_values = explainer([text])
    end_time = time.time()
    with open('./TIMING.csv', 'a') as time_file:
        time_file.write(f"shap,{output_filename},{end_time-start_time}\n")

    token_texts = shap_values.data[0].flatten()
    shap_scores = shap_values.values[0, :, 1]  # positive class

    df = pd.DataFrame({'index': token_texts, 'importance': shap_scores})
    imps_array = np.array(shap_scores)
    max_importance = np.max(np.abs(imps_array))
    df['scaled_importance'] = (imps_array * 100 / max_importance) if max_importance != 0 else 0
    df.to_csv(output_filepath, index=False)

# ---------- LIME Explanation ----------

def lime_explain_text_to_csv(text: str, output_filename: str, nsamps=100, overwrite=False):
    print(f'LIME: explaining text with {len(text)} characters')
    os.makedirs(f'DATA/LIME_TOKENS_{nsamps}', exist_ok=True)
    output_filepath = f'DATA/LIME_TOKENS_{nsamps}/{output_filename}.csv'
    if not overwrite and os.path.exists(output_filepath):
        return

    start_time = time.time()

    # LIME needs a prediction function returning probability array
    def lime_predict(texts):
        return predict_proba_batch(texts)

    explainer = LimeTextExplainer(class_names=['neg', 'pos'], random_state=3)
    explanation = explainer.explain_instance(text, lime_predict, num_features=100, num_samples=nsamps)
    end_time = time.time()

    words = []
    scores = []
    for word, score in explanation.as_list():
        words.append(word)
        scores.append(score)

    df = pd.DataFrame({'index': words, 'importance': scores})
    imps_array = np.array(scores)
    max_importance = np.max(np.abs(imps_array))
    df['scaled_importance'] = (imps_array * 100 / max_importance) if max_importance != 0 else 0
    df.to_csv(output_filepath, index=False)
    with open('./TIMING.csv', 'a') as time_file:
        time_file.write(f"lime,{output_filename},{end_time-start_time}\n")
    html_filepath = f'DATA/LIME_TOKENS_{nsamps}/{output_filename}.html'
    with open(html_filepath, 'w') as f:
        f.write(explanation.as_html())


class ModelWrapper(nn.Module):
    def __init__(self, model=model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

def ig_explain_text_to_csv(text: str, output_filename: str, overwrite=False):
    print(f'IG: explaining text with {len(text)} characters')
    os.makedirs('DATA/IG_TOKENS', exist_ok=True)
    output_filepath = f'DATA/IG_TOKENS/{output_filename}.csv'
    if not overwrite and os.path.exists(output_filepath):
        return
    
    start_time = time.time()
    
    # Initialize LayerIntegratedGradients with wrapped model
    wrapped_model = ModelWrapper(model)
    lig = LayerIntegratedGradients(wrapped_model, wrapped_model.model.get_input_embeddings())
    
    # Tokenize full text and process in chunks (same approach as predict_confidence)
    encoding = tokenizer(text, return_tensors='pt', truncation=False, padding=False)
    input_ids = encoding['input_ids'][0]
    max_len, stride = 510, 410
    total_len = len(input_ids)
    
    # Track attributions and counts for each token position
    token_attributions = {}
    token_counts = {}
    
    for start in range(0, total_len, stride):
        end = min(start + max_len, total_len)
        chunk_ids = input_ids[start:end]
        chunk = tokenizer.build_inputs_with_special_tokens(chunk_ids.tolist())
        input_tensor = torch.tensor([chunk])
        
        # Get attributions for this chunk
        chunk_attributions = lig.attribute(input_tensor, target=1)
        chunk_attr_scores = chunk_attributions.sum(dim=-1).squeeze(0).detach().numpy()
        
        # Map attributions back to original token positions
        # Skip special tokens at start/end when mapping back
        special_tokens_start = 1  # Skip CLS token
        special_tokens_end = 1    # Skip SEP token
        
        for i, attr_score in enumerate(chunk_attr_scores[special_tokens_start:-special_tokens_end]):
            original_pos = start + i
            if original_pos < total_len:
                if original_pos not in token_attributions:
                    token_attributions[original_pos] = 0
                    token_counts[original_pos] = 0
                token_attributions[original_pos] += attr_score
                token_counts[original_pos] += 1
    
    # Average attributions for overlapping tokens
    final_tokens = []
    final_attributions = []
    
    for pos in sorted(token_attributions.keys()):
        if pos < total_len:
            token = tokenizer.convert_ids_to_tokens([input_ids[pos]])[0]
            clean_token = tokenizer.convert_tokens_to_string([token]).strip()
            avg_attribution = token_attributions[pos] / token_counts[pos]
            final_tokens.append(clean_token)
            final_attributions.append(avg_attribution)
    
    end_time = time.time()
    with open('./TIMING.csv', 'a') as time_file:
        time_file.write(f"ig,{output_filename},{end_time-start_time}\n")
    
    # Create DataFrame
    df = pd.DataFrame({'index': final_tokens, 'importance': final_attributions})
    imps_array = np.array(final_attributions)
    max_importance = np.max(np.abs(imps_array))
    df['scaled_importance'] = (imps_array * 100 / max_importance) if max_importance != 0 else 0
    df.to_csv(output_filepath, index=False)

# ---------- Main Driver ----------

def main(nsamps=500, overwrite=False, lime=False, shap=False, ig=False, lime_subset_only=False):

    base_dir = 'DATA/SPAN_EXPS_stride_35_128_1e-05'
    print(f"Looking inside: {base_dir} for valid CSVs")
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    all_csv_files = []

    for subdir in sorted(subdirs):
        csvs = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".csv") and f != "meta.csv"]
        all_csv_files.extend(sorted(csvs))

    ids_to_test = []
    for csv_file in all_csv_files:
        csv_file = csv_file.split('/')[-1]
        csv_file = csv_file.split('.')[0]
        csv_file = csv_file.split('__')[0]
        ids_to_test.append(int(csv_file))

    # Optional speed/debug shortcut for LIME runs only.
    # This keeps the full dataset behavior unless explicitly requested.
    if lime and lime_subset_only:
        ids_to_test = [377, 405]
    # ids_to_test = [322, 377, 435, 56]
    # ids_to_test = [377]
    # ids_to_test = [405]
    
    print(f'Selected {len(ids_to_test)} case ids for processing.')
    print(f'Case ids to process: {ids_to_test}')

    input_df = pd.read_csv('./DATA/PREP/bert_input_sample.csv')
    for count, (uid, text) in enumerate(zip(input_df['Case Name'], input_df['Input']), 1):
        if count-1 not in ids_to_test:
            continue

        words = re.findall(r'[a-zA-Z0-9]+', uid)
        camel_cased = ''.join(word.capitalize() for word in words)
        filename = f'{count-1}__{camel_cased}'

        if shap: # run SHAP when requested
            shap_explain_text_to_csv(text, filename, nsamps=nsamps, overwrite=overwrite)

        if lime: # run LIME when requested
            lime_explain_text_to_csv(text, filename, nsamps=nsamps, overwrite=overwrite)
         
        if ig: # run IG
            ig_explain_text_to_csv(text, filename, overwrite=overwrite)

        if count % 20 == 1:
            print(f'Processed {count} of {len(input_df)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run explainers on inputs')
    parser.add_argument('--nsamps', type=int, default=500, help='number of samples for explainers (default: 500)')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing outputs (default: False)')
    parser.add_argument('--lime', action='store_true', help='run LIME explainer (default: False)')
    parser.add_argument(
        '--lime_subset_only',
        action='store_true',
        help='when used with --lime, restrict processing to the fixed subset ids [377, 405]'
    )
    parser.add_argument('--shap', action='store_true', help='run SHAP explainer (default: False)')
    parser.add_argument('--ig', action='store_true', help='run Integrated Gradients explainer (default: False)')
    args = parser.parse_args()

    main(
        nsamps=args.nsamps,
        overwrite=args.overwrite,
        lime=args.lime,
        shap=args.shap,
        ig=args.ig,
        lime_subset_only=args.lime_subset_only,
    )
