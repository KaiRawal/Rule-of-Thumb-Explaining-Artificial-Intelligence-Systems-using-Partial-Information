import os
import re
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer




parser = argparse.ArgumentParser(description='Merge token embeddings according to spans defined by a bitstring.')
parser.add_argument("--input_file", type=str, required=True, help=f"Path to input JSONL file")
parser.add_argument("--output_dir", type=str, required=True, help=f"Path to output dir")
args = parser.parse_args()

INPUT_CSV = args.input_file
MODEL_NAME = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
EMBEDDING_DIR = './DATA/EMBEDDINGS'
MERGED_DIR = f'{args.output_dir}'

# Load input
df = pd.read_csv(INPUT_CSV)


# Process each row
for count, row in tqdm(df.iterrows(), total=len(df)):
    uid = row['review_id']
    input_text = row['review_text']
    original_bitstring = row['bitstring']
    bitstring = original_bitstring

    embed_path = f'{EMBEDDING_DIR}/{uid}.csv'
    merged_path = f'{MERGED_DIR}/{uid}.csv'

    # if os.path.exists(merged_path):
    #     print(f"Already merged: {merged_path}")
    #     continue

    # if not os.path.exists(embed_path):
    #     if args.shap_samples == 0:
    #         print(f"Missing embedding file: {embed_path}")
    #     continue
    # else:
    #     if args.shap_samples > 0:
    #         print(f"Merging SHAP file: {embed_path}")
    
    # Load token-level embeddings
    token_df = pd.read_csv(embed_path, index_col=0, keep_default_na=False, na_values=[])
    token_df = token_df.iloc[1:-1]
    token_df.index = token_df.index.astype('str')
    tokens = token_df.index.tolist()
    embeddings = token_df.values

    # Tokenize to get offsets
    encoding = tokenizer(input_text, return_offsets_mapping=True, truncation=False, padding=False)
    offsets = encoding['offset_mapping'][1:-1]  # remove special tokens
    # token_ids = encoding['input_ids'][1:-1]
    # full_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Map each token to its bit (by offset alignment)
    token_bits = []
    for start, end in offsets:
        span_bits = bitstring[start:end]
        label = int(np.mean([int(c) if c else 0.5 for c in span_bits]) >= 0.5) if span_bits else 0
        token_bits.append(label)

    # Group contiguous tokens by bit
    spans = []
    current_label = token_bits[0]
    current_span_tokens = [tokens[0]]
    current_span_embs = [embeddings[0]]

    for i in range(1, len(tokens)):
        if token_bits[i] == current_label:
            current_span_tokens.append(tokens[i])
            current_span_embs.append(embeddings[i])
        else:
            span_str = " ".join([str(tok) for tok in current_span_tokens])
            if current_label == 1:
                span_str = f"___{span_str}"
            mean_emb = np.mean(np.stack(current_span_embs), axis=0)
            spans.append((span_str, mean_emb))
            current_label = token_bits[i]
            current_span_tokens = [tokens[i]]
            current_span_embs = [embeddings[i]]

    # Add final span
    if current_span_tokens:
        span_str = " ".join([str(tok) for tok in current_span_tokens])
        if current_label == 1:
            span_str = f"___{span_str}"
        mean_emb = np.mean(np.stack(current_span_embs), axis=0)
        spans.append((span_str, mean_emb))

    # Write output
    merged_df = pd.DataFrame([row[1] for row in spans], index=[row[0] for row in spans])
    merged_df.columns = [f'Dim_{i+1}' for i in range(merged_df.shape[1])]
    merged_df.to_csv(merged_path, index=True)