import numpy as np
import shap
import pandas as pd
import os
import re
import random
from tqdm import tqdm

TOKENS_DIR = './DATA/EMBEDDINGS_token_stride'
SPANS_DIR = './DATA/MERGED_span_stride_PredEx'

UNIFORM_TOKENS = './DATA/UNIFORM_tokens'
UNIFORM_SPANS = './DATA/UNIFORM_spans'
GAUSSIAN_TOKENS = './DATA/GAUSSIAN_tokens'
GAUSSIAN_SPANS = './DATA/GAUSSIAN_spans'

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


def generate_random(input_dir, output_dir, distribution='uniform'):
    csvs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]
    for file_name in tqdm(sorted(os.listdir(input_dir))):
        if not file_name.endswith('.csv'):
            continue
        csv_file = os.path.join(input_dir, file_name)
        token_df = pd.read_csv(csv_file, index_col=0)
        if 'token' in input_dir:
            token_df = token_df.iloc[1:-1]
        tokens = token_df.index.tolist()
        new_df = pd.DataFrame(index=tokens)
        if distribution.lower() == 'uniform':
            uniform_array = np.random.uniform(low=-1, high=1, size=len(tokens))
            new_df['importance'] = uniform_array
        elif distribution.lower() == 'gaussian':
            gaussian_array = np.random.normal(loc=0.0, scale=1.0, size=len(tokens))
            new_df['importance'] = gaussian_array
        new_df['scaled_importance'] = scale_importances(new_df['importance'], use_pow=5)
        new_df.to_csv(f'{output_dir}/{file_name}')

def main():
    np.random.seed(0)
    random.seed(0)
    # os.makedirs(UNIFORM_TOKENS, exist_ok=True)
    # os.makedirs(GAUSSIAN_TOKENS, exist_ok=True)
    os.makedirs(UNIFORM_SPANS, exist_ok=True)
    # os.makedirs(GAUSSIAN_SPANS, exist_ok=True)
    generate_random(SPANS_DIR, UNIFORM_SPANS, 'uniform')
    # generate_random(SPANS_DIR, GAUSSIAN_SPANS, 'gaussian')
    # generate_random(TOKENS_DIR, UNIFORM_TOKENS, 'uniform')
    # generate_random(TOKENS_DIR, GAUSSIAN_TOKENS, 'gaussian')
    pass

if __name__ == '__main__':
    main()
