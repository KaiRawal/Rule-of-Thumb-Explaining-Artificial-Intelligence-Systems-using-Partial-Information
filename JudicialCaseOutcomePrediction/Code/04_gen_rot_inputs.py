import argparse
import pandas as pd
import re
from sklearn.metrics import confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Generate input for RoT model with confusion matrix.")
    parser.add_argument('--mode', choices=['truncated', 'stride'], required=False, help="Prediction mode")
    args = parser.parse_args()

    # File paths
    input_file = './DATA/PREP/bert_input_sample.csv'
    if args.mode == 'truncated':
        print('truncated mode is deprecated')
        return
        pred_file = './DATA/model_preds_truncated.csv'
        output_file = './DATA/rot_input_embeddings_truncated.csv'
    else:
        pred_file = './DATA/PREP/model_preds_stride.csv'
        output_file = './DATA/PREP/rot_input_embeddings_stride.csv'

    # Read input and predictions
    df = pd.read_csv(input_file)
    # print(f"{df['Case Name'].iloc[79:84]=}")
    preds = pd.read_csv(pred_file)
    # print(f"{preds['Case Name'].iloc[79:84]=}")
    assert all(df['Case Name'] == preds['Case Name']), "Case Names don't align perfectly row-by-row"


    # case_name = "M/S Raptakos, Brett Vs. M/S Ganesh Property"
    # print(df['Case Name'].value_counts()[case_name])
    # print(preds['Case Name'].value_counts()[case_name])

    # Merge predictions back onto input
    # merged = df.merge(preds, on='Case Name', how='inner')
    merged = pd.concat([df, preds.drop(columns='Case Name')], axis=1)
    # print(f"{merged['Case Name'].iloc[79:84]=}")

    # Drop rows with missing predictions (optional: warn?)
    merged = merged.dropna(subset=['Model Probability'])

    # Derive binary label from prediction threshold 0.5
    merged['roberta_prediction'] = (merged['Model Probability'] >= 0.5).astype(int)
    merged['ground_truth_judgement'] = merged['Label'].astype(int)

    # Format the index
    merged['index'] = [
        f"{i}__{''.join(word.capitalize() for word in re.findall(r'[a-zA-Z0-9]+', cn))}"
        for i, cn in enumerate(merged['Case Name'])
    ]

    # Create transformed dataframe
    df_transformed = merged[['index', 'roberta_prediction', 'ground_truth_judgement']] # 
    df_transformed.to_csv(output_file, index=True)

    # Compute confusion matrix
    y_true = df_transformed['ground_truth_judgement']
    y_pred = df_transformed['roberta_prediction']
    cm = confusion_matrix(y_true, y_pred)
    print(sum(y_true))
    print(sum(y_pred))
    print(len(y_true))
    print(len(y_pred))
    

    print("\nConfusion Matrix (Actual vs Predicted):")
    print("          Predicted 0    Predicted 1")
    print(f"Actual 0     {cm[0][0]:>5}           {cm[0][1]:>5}")
    print(f"Actual 1     {cm[1][0]:>5}           {cm[1][1]:>5}")

if __name__ == '__main__':
    main()
