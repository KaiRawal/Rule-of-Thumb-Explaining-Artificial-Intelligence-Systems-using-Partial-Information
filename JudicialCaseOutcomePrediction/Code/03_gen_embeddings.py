import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import time

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("L-NLProc/PredEx_RoBERTa_Large_Pred")
model = AutoModelForSequenceClassification.from_pretrained("L-NLProc/PredEx_RoBERTa_Large_Pred", trust_remote_code=True)
model.eval()

def filter_df(infile='./DATA/PREP/subset.csv', outfile='./DATA/PREP/bert_input_sample.csv'):
    df = pd.read_csv(infile)
    filtered_df = df[['Case Name', 'Input', 'Output', 'Label', 'bitstring']]
    filtered_df = filtered_df.dropna()
    filtered_df.to_csv(outfile, index=False)

def embed_and_predict_truncated(input_passage='', output_file_path=''):
    if os.path.exists(output_file_path):
        return None

    inputs = tokenizer(input_passage, return_tensors='pt', truncation=True, padding=True)
    inputs['output_hidden_states'] = True

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence_score = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()

    embeddings = outputs.hidden_states[-1][0]  # shape: (seq_len, hidden_dim)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    embeddings_list = embeddings.cpu().numpy()

    if np.isnan(embeddings_list).any():
        print(f'NANs in {output_file_path}')

    clean_tokens = [tokenizer.convert_tokens_to_string([t]).strip() for t in tokens]
    df = pd.DataFrame(embeddings_list, index=clean_tokens)
    heads = [f'Dim_{i+1}' for i in range(df.shape[1])]
    df.columns = heads
    df.to_csv(output_file_path, index=True)

    return confidence_score

def embed_and_predict_stride(input_passage='', output_file_path=''):
    if os.path.exists(output_file_path):
        print(f'Skipping {output_file_path}')
        return None
    
    start_time = time.time()

    encoding = tokenizer(
        input_passage,
        return_offsets_mapping=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=False,
        padding=False
    )
    input_ids = encoding['input_ids'][0]
    full_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    max_len = 510
    stride = 410
    total_len = len(input_ids)
    seen_token_indices = set()
    rows = []
    confidence_scores = []

    for start in range(0, total_len, stride):
        end = min(start + max_len, total_len)
        chunk_ids = input_ids[start:end]
        chunk = tokenizer.build_inputs_with_special_tokens(chunk_ids.tolist())
        input_tensor = torch.tensor([chunk])
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            outputs = model(input_tensor, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()
            confidence_scores.append(confidence)

            embeddings = outputs.hidden_states[-1][0]
            chunk_tokens = tokenizer.convert_ids_to_tokens(chunk)

        valid_range = slice(1, -1) if len(chunk_tokens) > 2 else slice(0, len(chunk_tokens))
        token_embeddings = embeddings[valid_range]
        tokens_in_chunk = chunk_tokens[valid_range]

        for i, tok in enumerate(tokens_in_chunk):
            orig_idx = start + i
            if orig_idx in seen_token_indices or orig_idx >= len(full_tokens):
                continue
            seen_token_indices.add(orig_idx)
            emb = token_embeddings[i].cpu().numpy()
            rows.append((tokenizer.convert_tokens_to_string([full_tokens[orig_idx]]).strip(), *emb))
    
    end_time = time.time()

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.set_index(0, inplace=True)
    df.columns = [f'Dim_{i+1}' for i in range(df.shape[1])]
    df.to_csv(output_file_path, index=True)

    return np.mean(confidence_scores), end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings with truncation or stride mode.")
    parser.add_argument('--mode', choices=['truncated', 'stride'], required=False, help="Embedding mode")
    args = parser.parse_args()

    filter_df()  # always uses subset.csv and produces bert_input_sample.csv

    input_df = pd.read_csv('./DATA/PREP/bert_input_sample.csv')

    if args.mode == 'truncated':
        print('Truncated mode is deprecated')
        return
        embedding_dir = './DATA/EMBEDDINGS_token_truncated/'
        prediction_file = './DATA/model_preds_truncated.csv'
        embedding_fn = embed_and_predict_truncated
    else:
        embedding_dir = './DATA/EMBEDDINGS_token_stride/'
        prediction_file = './DATA/PREP/model_preds_stride.csv'
        embedding_fn = embed_and_predict_stride

    os.makedirs(embedding_dir, exist_ok=True)

    results = []

    for count, (uid, text) in enumerate(zip(input_df['Case Name'], input_df['Input']), 1):
        words = re.findall(r'[a-zA-Z0-9]+', uid)
        camel_cased = ''.join(word.capitalize() for word in words)
        filename = f'{count-1}__{camel_cased}.csv'
        out_path = os.path.join(embedding_dir, filename)
        prob, time_taken = embedding_fn(input_passage=text, output_file_path=out_path)
        results.append({'Case Name': uid, 'Model Probability': prob})
        with open('./TIMING.csv', 'a') as time_file:
            time_file.write(f"embedding_generation,{filename[:-4]},{time_taken}\n")


        if count % 20 == 1:
            print(f'Processed {count} of {len(input_df)}')

    pd.DataFrame(results).to_csv(prediction_file, index=False)

if __name__ == '__main__':
    main()
