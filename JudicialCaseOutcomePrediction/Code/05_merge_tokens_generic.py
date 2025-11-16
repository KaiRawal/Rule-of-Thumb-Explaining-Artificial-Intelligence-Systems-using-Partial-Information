import os
import re
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer




# Parse command line arguments
parser = argparse.ArgumentParser(description='Merge token embeddings according to spans defined by a bitstring.')
parser.add_argument('--num_spans', type=int, help='Number of alternations in the bitstring (overrides default bitstring)')
parser.add_argument('--shap_samples', type=int, default=0, help='If positive, use SHAP token embeddings with the given number of samples')
parser.add_argument('--ig_dir', type=str, default='', help='If specified, use IG token embeddings from specified directory')
parser.add_argument('--random_dir', type=str, default='', help='If exists, merge the random data')

args = parser.parse_args()

# Constants
INPUT_CSV = './DATA/PREP/bert_input_sample.csv'
MODEL_NAME = "L-NLProc/PredEx_RoBERTa_Large_Pred"

# Default directories
EMBEDDING_DIR = './DATA/EMBEDDINGS_token_stride'
MERGED_DIR = './DATA/MERGED_span_stride_PredEx'

if len(args.random_dir) > 0:
    assert args.shap_samples == 0
    assert args.num_spans is None
    assert args.ig_dir == ''
    EMBEDDING_DIR = f'./DATA/{args.random_dir}'
    MERGED_DIR = f'./DATA/{args.random_dir}_MERGED'


# Override for SHAP mode
if args.shap_samples > 0:
    assert args.num_spans is None
    assert args.ig_dir == ''
    EMBEDDING_DIR = f'./DATA/SHAP_TOKENS_{args.shap_samples}'
    MERGED_DIR = f'./DATA/SHAP_MERGED_{args.shap_samples}'

# Override for IG mode
if args.ig_dir != '':
    assert args.num_spans is None
    assert args.shap_samples == 0
    EMBEDDING_DIR = f'./DATA/{args.ig_dir}_TOKENS'
    MERGED_DIR = f'./DATA/{args.ig_dir}_MERGED'

# Override for fixed number of spans
if args.num_spans is not None:
    MERGED_DIR = f'./DATA/MERGED_span_stride_fixed_{args.num_spans}'



# Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
os.makedirs(MERGED_DIR, exist_ok=True)

# Load input
df = pd.read_csv(INPUT_CSV)

def scale_importances(importances, use_logarithm=False, use_pow=-1):
    """Scale importance values for visualization."""
    if use_logarithm:
        importances = np.sign(importances) * np.log1p(np.abs(importances))
        # max_importance = np.max(np.abs(importances))
    elif use_pow > 0:
        importances = np.power(importances, use_pow)
    else:
        pass
    max_importance = np.max(np.abs(importances))

    importances = importances * 100 / max_importance
    return importances


def create_alternating_bitstring(length, num_alternations):
    """Create alternating bitstring with specified number of alternations."""
    if num_alternations <= 0:
        return '0' * length
    
    # Calculate segment size
    segment_size = length // num_alternations
    remainder = length % num_alternations
    
    bitstring = ""
    current_bit = '0'
    
    for i in range(num_alternations):
        # Add extra character to first segments if there's remainder
        extra = 1 if i < remainder else 0
        segment_length = segment_size + extra
        
        bitstring += current_bit * segment_length
        # Alternate between '0' and '1'
        current_bit = '1' if current_bit == '0' else '0'
    
    return bitstring

# Process each row
for count, row in tqdm(df.iterrows(), total=len(df)):
    uid = row['Case Name']
    input_text = row['Input']
    original_bitstring = row['bitstring']
    bitstring = original_bitstring
    
    # Override bitstring if stride is specified
    if args.num_spans is not None:
        bitstring = create_alternating_bitstring(len(original_bitstring), args.num_spans)

    # Build filename
    words = re.findall(r'[a-zA-Z0-9]+', uid)
    camel_cased = ''.join(word.capitalize() for word in words)
    filename = f'{count}__{camel_cased}.csv'
    embed_path = os.path.join(EMBEDDING_DIR, filename)
    merged_path = os.path.join(MERGED_DIR, filename)

    if os.path.exists(merged_path):
        print(f"Already merged: {merged_path}")
        continue

    if not os.path.exists(embed_path):
        if args.shap_samples == 0:
            print(f"Missing embedding file: {embed_path}")
        continue
    else:
        if args.shap_samples > 0:
            print(f"Merging SHAP file: {embed_path}")
    
    # Load token-level embeddings
    token_df = pd.read_csv(embed_path, index_col=0)
    token_df = token_df.iloc[1:-1]
    tokens = token_df.index.tolist()
    embeddings = token_df.values

    # Tokenize to get offsets
    encoding = tokenizer(input_text, return_offsets_mapping=True, truncation=False, padding=False)
    offsets = encoding['offset_mapping'][1:-1]  # remove special tokens
    token_ids = encoding['input_ids'][1:-1]
    full_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Map each token to its bit (by offset alignment)
    token_bits = []
    for start, end in offsets:
        span_bits = bitstring[start:end]
        label = int(span_bits[0]) if span_bits else 0
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
    merged_df = pd.DataFrame([row[1] for row in spans],
                             index=[row[0] for row in spans])
    merged_df.columns = [f'Dim_{i+1}' for i in range(merged_df.shape[1])]
    if args.shap_samples > 0 or len(args.random_dir) > 0 or args.ig_dir != '':
        merged_df = merged_df.rename(columns={'Dim_1': 'importance', 'Dim_2': 'scaled_importance'})
        # merged_df['scaled_importance'] = scale_importances(merged_df['importance'], use_pow=5)
        merged_df['scaled_importance'] = scale_importances(merged_df['importance'])
    merged_df.to_csv(merged_path, index=True)