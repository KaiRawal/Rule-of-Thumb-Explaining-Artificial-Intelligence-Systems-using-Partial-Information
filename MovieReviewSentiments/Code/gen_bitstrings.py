import json
import pandas as pd
import numpy as np
import re
import string
import argparse
from tqdm import tqdm

GPT_INPUT_FILE_train = './DATA/gpt/train_predictions.jsonl'
GPT_INPUT_FILE_test = './DATA/gpt/test_predictions.jsonl'
GPT_INPUT_FILE_val = './DATA/gpt/val_predictions.jsonl'
HUMAN_INPUT_FILE_train = './DATA/movies/train.jsonl'
HUMAN_INPUT_FILE_test = './DATA/movies/test.jsonl.bak'
HUMAN_INPUT_FILE_val = './DATA/movies/val.jsonl'

DOCS_DIR = './DATA/movies/docs'


def normalize_with_mapping(text):
    """Lowercase text, strip spaces/punctuation, track mapping to original indices."""
    norm_chars = []
    mapping = []
    for i, ch in enumerate(text):
        if ch.lower() not in string.whitespace + string.punctuation:
            norm_chars.append(ch.lower())
            mapping.append(i)
    return "".join(norm_chars), mapping

def review_bitmask(review, evidences):
    norm_review, mapping = normalize_with_mapping(review)
    bitmask = ["0"] * len(review)

    for ev in evidences:
        norm_ev, _ = normalize_with_mapping(ev)
        if not norm_ev:
            continue

        # manual traversal
        i = 0
        while i <= len(norm_review) - len(norm_ev):
            if norm_review[i:i+len(norm_ev)] == norm_ev:
                orig_start = mapping[i]
                orig_end = mapping[i + len(norm_ev) - 1]
                for j in range(orig_start, orig_end + 1):
                    bitmask[j] = "1"
                break  # stop after first match for this evidence
            i += 1

    return "".join(bitmask)



def get_bits(inputfiles):
    all_reviews = []
    all_bitstrings = []
    all_ids = []
    for input_file in inputfiles:
        print(f'Matching evidences for reviews from {input_file}')
        with open(input_file, "r", encoding="utf-8") as fin:
            for line in tqdm(fin):
                entry = json.loads(line)
                annotation_id = entry["annotation_id"]
                doc_path = f'{DOCS_DIR}/{annotation_id}'
                with open(doc_path, "r", encoding="utf-8") as dfile:
                    review_text = dfile.read().strip()
                evidence_list = None
                if "evidence" in entry:
                    evidence_list = entry["evidence"]
                elif "evidences" in entry:
                    if len(entry['evidences']) == 0:
                        continue
                    evidence_list = [ev[0]['text'] for ev in entry["evidences"]]
                else:
                    pass
                bitstring = review_bitmask(review_text, evidence_list)
                all_reviews.append(review_text)
                all_bitstrings.append(bitstring)
                all_ids.append(annotation_id[:-4])
    return all_ids, all_reviews, all_bitstrings


def main():
    ids, reviews, bits = get_bits([GPT_INPUT_FILE_train])
    df_gpt_train = pd.DataFrame({'review_id': ids, 'review_text': reviews, 'bitstring': bits})
    df_gpt_train.to_csv('./DATA/CSVs/gpt_train.csv')

    # ids, reviews, bits = get_bits([GPT_INPUT_FILE_test, GPT_INPUT_FILE_val])
    # df_gpt_test = pd.DataFrame({'review_id': ids, 'review_text': reviews, 'bitstring': bits})
    # df_gpt_test.to_csv('./DATA/CSVs/gpt_test.csv')
    
    ids, reviews, bits = get_bits([HUMAN_INPUT_FILE_train])
    df_human_train = pd.DataFrame({'review_id': ids, 'review_text': reviews, 'bitstring': bits})
    df_human_train.to_csv('./DATA/CSVs/human_train.csv')
    
    # ids, reviews, bits = get_bits([HUMAN_INPUT_FILE_test, HUMAN_INPUT_FILE_val])
    # df_human_test = pd.DataFrame({'review_id': ids, 'review_text': reviews, 'bitstring': bits})
    # df_human_test.to_csv('./DATA/CSVs/human_test.csv')



example_json_string = """{"annotation_id": "negR_003.txt", "classification": "NEG", "evidences": [[{"docid": "negR_003.txt", "end_sentence": 4, "end_token": 117, "start_sentence": 3, "start_token": 114, "text": "dead on arrival"}], [{"docid": "negR_003.txt", "end_sentence": 17, "end_token": 389, "start_sentence": 16, "start_token": 386, "text": "the characters stink"}], [{"docid": "negR_003.txt", "end_sentence": 12, "end_token": 357, "start_sentence": 11, "start_token": 345, "text": "subpar animation , instantly forgettable songs , poorly - integrated computerized footage"}], [{"docid": "negR_003.txt", "end_sentence": 23, "end_token": 588, "start_sentence": 22, "start_token": 584, "text": "complete lack of personality"}], [{"docid": "negR_003.txt", "end_sentence": 10, "end_token": 306, "start_sentence": 9, "start_token": 303, "text": "missing pure showmanship"}], [{"docid": "negR_003.txt", "end_sentence": 22, "end_token": 569, "start_sentence": 21, "start_token": 563, "text": "will probably be as bored watching"}], [{"docid": "negR_003.txt", "end_sentence": 24, "end_token": 597, "start_sentence": 23, "start_token": 595, "text": "this mess"}], [{"docid": "negR_003.txt", "end_sentence": 22, "end_token": 552, "start_sentence": 21, "start_token": 543, "text": "one must strain through too much of this mess"}], [{"docid": "negR_003.txt", "end_sentence": 6, "end_token": 147, "start_sentence": 5, "start_token": 140, "text": "is n't nearly as dull as this"}]], "query": "What is the sentiment of this review?", "query_type": null}"""

def test():
    entry = json.loads(example_json_string)
    print(entry)
    annotation_id = entry["annotation_id"]
    doc_path = f'{DOCS_DIR}/{annotation_id}'
    with open(doc_path, "r", encoding="utf-8") as dfile:
        review_text = dfile.read().strip()
    evidence_list = None
    if "evidence" in entry:
        evidence_list = entry["evidence"]
    elif "evidences" in entry:
        if len(entry['evidences']) == 0:
            return
        evidence_list = [ev[0]['text'] for ev in entry["evidences"]]
    else:
        pass
    print(review_text)
    print(evidence_list)
    print(review_bitmask(review_text, evidence_list))


if __name__ == '__main__':
    main()
    # test()
