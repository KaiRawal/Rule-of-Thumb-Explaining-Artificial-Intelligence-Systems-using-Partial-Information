from transformers import BertTokenizer, BertModel
from transformers import LongformerTokenizer, LongformerModel
from transformers import BigBirdTokenizer, BigBirdModel
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm

DEFAULT_INPUT_FILE = './DATA/gpt/train_predictions.jsonl'
DOCS_DIR = './DATA/movies/docs'


tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
model = AutoModel.from_pretrained('answerdotai/ModernBERT-base')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', force_download=True)
# model = LongformerModel.from_pretrained('allenai/longformer-base-4096', force_download=True)
# tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base', force_download=True)
# model = BigBirdModel.from_pretrained('google/bigbird-roberta-base', force_download=True)

model.eval()




def embed_and_save(input_passage='', output_file_name='empty'):
    inputs = tokenizer(input_passage, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    assert len(tokens) < 8192 # ModernBERT length
    embeddings_list = embeddings[0].cpu().numpy()
    if np.isnan(embeddings_list).any():
        print(f'NANs in {output_file_name}')
    clean_tokens = [tokenizer.convert_tokens_to_string([t]).strip() for t in tokens]
    df = pd.DataFrame(embeddings_list, index=clean_tokens)
    heads = [f'Dim_{i+1}' for i in range(df.shape[1])]
    df.columns = heads
    df.to_csv(f'./DATA/EMBEDDINGS/{output_file_name}.csv', index=True)
    return df.mean().values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE, help=f"Path to input JSONL file (default: {DEFAULT_INPUT_FILE})")
    args = parser.parse_args()
    print(f'embedding all reviews from {args.input_file}')
    with open(args.input_file, "r", encoding="utf-8") as fin:
        for line in tqdm(fin):
            entry = json.loads(line)
            annotation_id = entry["annotation_id"]
            doc_path = f'{DOCS_DIR}/{annotation_id}'
            with open(doc_path, "r", encoding="utf-8") as dfile:
                review_text = dfile.read().strip()
            embed_and_save(review_text, annotation_id[:-4])


if __name__ == '__main__':
    main()


