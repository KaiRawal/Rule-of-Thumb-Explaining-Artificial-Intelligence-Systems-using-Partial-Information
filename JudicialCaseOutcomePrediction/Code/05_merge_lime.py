import os
import re
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Merge token embeddings according to spans defined by a bitstring.')
parser.add_argument('--lime_samples', type=int, required=True, help='Specify LIME input directory')
args = parser.parse_args()

# Constants
INPUT_CSV = './DATA/PREP/bert_input_sample.csv'
EMBEDDING_DIR = f'./DATA/LIME_tokens_{args.lime_samples}'
MERGED_DIR = f'./DATA/LIME_MERGED_{args.lime_samples}'


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


def split_case(text, bitstring):
    result = []
    previous_bit = None
    for char, bit in zip(text, bitstring):
        if len(result) > 0 and bit == previous_bit:
            result[-1] += char
        else:
            if bit == '0':
                result.append(f'{char}')
            else:
                result.append(f'___{char}')
            previous_bit = bit
    result = [r.strip() for r in result]
    final_result = [result[0]]
    prev_bit = final_result[-1].startswith('___')
    for r in result[1:]:
        if r == '___' or r == '':
            continue
        if len(r) == 1:
            print(f'{r} \t {result=}')
        if r.startswith('___') and prev_bit:
            final_result[-1] += r[3:]
        elif not r.startswith('___') and not prev_bit:
            final_result[-1] += r
        else:
            final_result.append(r)
            prev_bit = final_result[-1].startswith('___')
    return final_result


def get_average_importance(span, token2importance, ignore_missing=False):
    if span.startswith('___'):
        span = span[3:]
    tokenized = re.split(r'\W+', span)
    tokenized = [t for t in tokenized if len(t.strip()) > 0]
    span_imps = []
    if ignore_missing:
        span_imps = [token2importance[t] for t in tokenized if t in token2importance]
    else:
        span_imps = [token2importance.get(t, 0) for t in tokenized]
    if sum(span_imps) == 0:
        return 0
    return sum(span_imps) / len(span_imps)


# Process each row
for count, row in tqdm(df.iterrows(), total=len(df)):
    uid = row['Case Name']
    input_text = row['Input']
    original_bitstring = row['bitstring']
    bitstring = original_bitstring
    

    # Build filename
    words = re.findall(r'[a-zA-Z0-9]+', uid)
    camel_cased = ''.join(word.capitalize() for word in words)
    filename = f'{count}__{camel_cased}.csv'
    embed_path = os.path.join(EMBEDDING_DIR, filename)
    merged_path = os.path.join(MERGED_DIR, filename)

    if os.path.exists(merged_path):
        pass
        # print(f"Already averaged span importances in output file: {merged_path}")
        # continue

    if not os.path.exists(embed_path):
        # print(f"Missing importances input file: {embed_path}")
        continue
    
    # Load token-level embeddings
    token_df = pd.read_csv(embed_path, index_col=0)
    # token_df = token_df.iloc[1:-1]
    tokens = token_df.index.tolist()
    importances = token_df['scaled_importance'].values
    tok2imp = {tok: imp for tok, imp in zip(tokens, importances)}

    spans = split_case(input_text, bitstring)
    raw_importance = [get_average_importance(span, tok2imp) for span in spans]

    merged_df = pd.DataFrame([imp for imp in raw_importance],
                             index=[row for row in spans])
    merged_df.columns = [f'Dim_{i+1}' for i in range(merged_df.shape[1])]
    merged_df = merged_df.rename(columns={'Dim_1': 'importance'})
    # merged_df['scaled_importance'] = scale_importances(merged_df['importance'], use_pow=5)
    merged_df['scaled_importance'] = scale_importances(merged_df['importance'])
    merged_df.to_csv(merged_path, index=True)