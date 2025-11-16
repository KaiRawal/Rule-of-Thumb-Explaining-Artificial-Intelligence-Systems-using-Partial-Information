import json
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


GPT_INPUT_FILE_train = './DATA/gpt/train_predictions.jsonl'
GPT_INPUT_FILE_test = './DATA/gpt/test_predictions.jsonl'
GPT_INPUT_FILE_val = './DATA/gpt/val_predictions.jsonl'
HUMAN_INPUT_FILE_train = './DATA/movies/train.jsonl'
HUMAN_INPUT_FILE_test = './DATA/movies/test.jsonl.bak'
HUMAN_INPUT_FILE_val = './DATA/movies/val.jsonl'

DOCS_DIR = './DATA/movies/docs'

def get_human_output(review_id):
    with open(HUMAN_INPUT_FILE_test, "r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            if entry["annotation_id"][:-4] == review_id:
                if entry["classification"] == 'POS':
                    return 1
                else:
                    return 0
    with open(HUMAN_INPUT_FILE_val, "r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            if entry["annotation_id"][:-4] == review_id:
                if entry["classification"] == 'POS':
                    return 1
                else:
                    return 0
    return -1

def get_gpt_output(review_id):
    with open(GPT_INPUT_FILE_test, "r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            if entry["annotation_id"][:-4] == review_id:
                if entry["prediction"] == 'POS':
                    return 1
                else:
                    return 0
    with open(GPT_INPUT_FILE_val, "r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            if entry["annotation_id"][:-4] == review_id:
                if entry["prediction"] == 'POS':
                    return 1
                else:
                    return 0
    return -1

def produce_df(jsonl_files):
    all_lengths = []
    count = 0

    all_ids = []
    all_token_files = []
    all_merged_files = []
    all_num_evidences = []
    all_avg_evidence_lengths = []
    all_total_review_lengths = []
    all_PREDICTIONs = []
    human_predictions = []
    gpt_predictions = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as fin:
            for line in tqdm(fin):
                entry = json.loads(line)
                annotation_id = entry["annotation_id"][:-4]
                doc_path = f'{DOCS_DIR}/{annotation_id}.txt'
                with open(doc_path, "r", encoding="utf-8") as dfile:
                    review_text = dfile.read().strip()
                evidence_list = None
                output = None
                mfile = None
                if "evidence" in entry:
                    evidence_list = entry["evidence"]
                    output = 1 if entry["prediction"] == 'POS' else 0
                    mfile = f'./DATA/MERGED_GPT/{annotation_id}.csv'
                elif "evidences" in entry:
                    if len(entry['evidences']) == 0:
                        continue
                    evidence_list = [ev[0]['text'] for ev in entry["evidences"]]
                    output = 1 if entry["classification"] == 'POS' else 0
                    mfile = f'./DATA/MERGED_HUMAN/{annotation_id}.csv'
                else:
                    pass
                tfile = f'./DATA/EMBEDDINGS/{annotation_id}.csv'
                tok_df = pd.read_csv(tfile)
                all_lengths.append(len(tok_df))
                # if len(tok_df) > 1430:  # DROP LONG REVIEWS
                #     count += 1
                #     # print(f'\t{count}\t{annotation_id=} \t {len(tok_df)=}')
                #     continue
                all_ids.append(annotation_id)
                all_token_files.append(tfile)
                all_merged_files.append(mfile)
                all_num_evidences.append(len(evidence_list))
                all_avg_evidence_lengths.append(sum([len(ev) for ev in evidence_list]) / len(evidence_list))
                all_total_review_lengths.append(len(review_text))
                all_PREDICTIONs.append(output)
                human_predictions.append(get_human_output(annotation_id))
                gpt_predictions.append(get_gpt_output(annotation_id))
    # bins = range(min(all_lengths), max(all_lengths) + 2)
    # hist, bin_edges = np.histogram(all_lengths, bins=bins)
    # cumsum = np.cumsum(hist)
    # plt.step(bin_edges[:-1], cumsum, where='mid', marker=',')
    # plt.show()
    # plt.hist(all_lengths, bins=range(min(all_lengths),2+max(all_lengths)), edgecolor='black', align='left')
    return pd.DataFrame({'review_id': all_ids, 'token_embedding_file': all_token_files, 'merged_embedding_file': all_merged_files, 'num_evidences': all_num_evidences, 'avg_evidence_length': all_avg_evidence_lengths, 'review_length': all_total_review_lengths, 'rot_target': all_PREDICTIONs, 'human_prediction': human_predictions, 'gpt_prediction': gpt_predictions})
    pass

def main(inputs, output):
    df = produce_df(inputs)
    df.to_csv(output)
    print(inputs)
    print(df.head())
    print(df.describe())
    print(output)
    print()
    print()
    print()
    print()

if __name__ == '__main__':
    main(inputs=[GPT_INPUT_FILE_train], output='./DATA/CSVs/rot_input_gpt_train.csv')
    # main(inputs=[GPT_INPUT_FILE_test, GPT_INPUT_FILE_val], output='./DATA/CSVs/rot_input_gpt_test.csv')
    main(inputs=[HUMAN_INPUT_FILE_train], output='./DATA/CSVs/rot_input_human_train.csv')
    # main(inputs=[HUMAN_INPUT_FILE_test, HUMAN_INPUT_FILE_val], output='./DATA/CSVs/rot_input_human_test.csv')
    pass

