import argparse
import json
from pathlib import Path
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
openai.api_key = os.environ["openai_api_key"]

DEFAULT_INPUT_FILE = Path("./DATA/movies/train.jsonl")
DEFAULT_OUTPUT_FILE = Path("./DATA/gpt/train_predictions.jsonl")
DOCS_DIR = Path("./DATA/movies/docs")

# Retry wrapper for network issues
@retry(stop=stop_after_attempt(10), wait=wait_exponential(min=1, max=30))
def classify_review(review_text: str):
    response = openai.ChatCompletion.create(
        model="gpt-4.1-nano",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a sentiment classifier."},
            {"role": "user", "content": f"""Classify the sentiment of the following movie review.
Return JSON with exactly two keys:
- "prediction": either "POS" or "NEG".
- "evidence": a list of verbatim quotes from the text that justify the prediction.

Review:
{review_text}
"""
            }
        ]
    )
    content = response["choices"][0]["message"]["content"]

    return json.loads(content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Path to input JSONL file (default: ./DATA/movies/train.jsonl)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Path to output JSONL file (default: ./DATA/gpt/train_predictions.jsonl)")
    args = parser.parse_args()
    count = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            annotation_id = entry["annotation_id"]
            doc_path = DOCS_DIR / annotation_id
            if not doc_path.exists():
                print(f"Warning: missing review file {doc_path}")
                continue
            
            with doc_path.open("r", encoding="utf-8") as dfile:
                review_text = dfile.read().strip()
            
            try:
                result = classify_review(review_text)
            except Exception as e:
                print(f"Failed to classify {annotation_id}: {e}")
                continue
            
            out_entry = {
                "annotation_id": annotation_id,
                "prediction": result.get("prediction"),
                "evidence": result.get("evidence"),
            }
            count += 1
            print(f"{count} \t {annotation_id=} \t {out_entry['prediction']=}")
            fout.write(json.dumps(out_entry) + "\n")

if __name__ == "__main__":
    main()
