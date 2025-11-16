import pandas as pd
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from rule_of_thumb import RuleOfThumb
import argparse
import sys
import copy
from tqdm import tqdm
import time

# Global constants
DATA_DIR = './DATA'

# Inputs
INPUT_CSV = f'{DATA_DIR}/PREP/rot_input_embeddings_stride.csv'
SCORE_CSV = INPUT_CSV #'DATA/PREP/rot_input_embeddings_stride.csv'
TEST_MERGED_SPAN_DIR = f'{DATA_DIR}/MERGED_span_stride_PredEx'
# TRAIN_MERGED_SPAN_DIR = None

# Cache Suffixes
XX_FILE = f'xx.npy'
YY_FILE = f'yy.npy'
ESSAY_IDS_FILE = f'essay_ids.pkl'
TOKENS_BY_ID_FILE = f'tokens_by_id.pkl'

# Outputs
MODEL_FILENAME = 'rot_model.pkl'
PERFORMANCE_FILENAME = 'model_performance.txt'
HYPERPARAMS_FILENAME = 'hyperparams.txt'
META_CSV_FILENAME = 'meta.csv'
DATASET_VISUALISED_FILENAME = 'DATASET_VISUALISED.txt'

torch.set_num_threads(torch.get_num_interop_threads())

def parse_arguments():
    """Parse command line arguments for ROT model training."""
    parser = argparse.ArgumentParser(description='Training Rule of Thumb model')
    parser.add_argument('--epochs', type=int, default=35, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--l1_penalty', type=float, default=0, help='L1 penalty')
    parser.add_argument('--use_test', type=bool, default=True, help='Use test set')
    parser.add_argument('--train_set', type=str, required=True, help='Which Directory are the Training Spans in?')
    return parser.parse_args()

# def initialise_constants(args):
#     global TRAIN_MERGED_SPAN_DIR
#     # global XX_FILE
#     # global YY_FILE
#     # global ESSAY_IDS_FILE
#     # global TOKENS_BY_ID_FILE
#     # # global MODEL_FILENAME
#     # # global PERFORMANCE_FILENAME
#     # # global HYPERPARAMS_FILENAME
#     # # global META_CSV_FILENAME
#     # # global DATASET_VISUALISED_FILENAME
#     TRAIN_MERGED_SPAN_DIR = args.train_set
#     # cache_dir = TRAIN_MERGED_SPAN_DIR
#     # XX_FILE = f'{DATA_DIR}/{cache_dir}/xx.npy'
#     # YY_FILE = f'{DATA_DIR}/{cache_dir}/yy.npy'
#     # ESSAY_IDS_FILE = f'{DATA_DIR}/{cache_dir}/essay_ids.pkl'
#     # TOKENS_BY_ID_FILE = f'{DATA_DIR}/{cache_dir}/tokens_by_id.pkl'
#     # # MODEL_FILENAME = 'rot_model.pkl'
#     # # PERFORMANCE_FILENAME = 'model_performance.txt'
#     # # HYPERPARAMS_FILENAME = 'hyperparams.txt'
#     # # META_CSV_FILENAME = 'meta.csv'
#     # # DATASET_VISUALISED_FILENAME = 'DATASET_VISUALISED.txt'


def load_initial_data():
    """Load the initial CSV data and extract essay IDs."""
    df = pd.read_csv(INPUT_CSV)
    df = df.drop(df.columns[0], axis=1)
    essay_ids = list(df['index'])
    return df, essay_ids


def load_or_process_embeddings(
    all_essay_ids,
    merged_span_dir
):
    """Load embeddings from cache or process them from individual CSV files."""
    xx_file = f'{merged_span_dir}/{XX_FILE}'
    yy_file = f'{merged_span_dir}/{YY_FILE}'
    essay_ids_file = f'{merged_span_dir}/{ESSAY_IDS_FILE}'
    tokens_by_id_file = f'{merged_span_dir}/{TOKENS_BY_ID_FILE}'
    if (os.path.exists(xx_file) and os.path.exists(yy_file) and 
        os.path.exists(essay_ids_file) and os.path.exists(tokens_by_id_file)):
        
        # Load from disk
        xx = np.load(xx_file)
        yy = np.load(yy_file)
        with open(essay_ids_file, 'rb') as f:
            essay_ids = pickle.load(f)
        with open(tokens_by_id_file, 'rb') as f:
            tokens_by_id = pickle.load(f)
        print(f"Loaded xx, yy, essay_ids, and tokens_by_id from {merged_span_dir}.")
        
    else:
        # Process and save to disk
        xx, yy, essay_ids, tokens_by_id = process_embeddings_from_csv(
            all_essay_ids,
            merged_span_dir
        )
        
        # Save to disk
        np.save(xx_file, xx)
        np.save(yy_file, yy)
        with open(essay_ids_file, 'wb') as f:
            pickle.dump(essay_ids, f)
        with open(tokens_by_id_file, 'wb') as f:
            pickle.dump(tokens_by_id, f)
        print(f"Saved xx, yy, essay_ids, and tokens_by_id to {merged_span_dir}.")
    
    return xx, yy, essay_ids, tokens_by_id


def process_embeddings_from_csv(essay_ids, merged_span_dir):
    """Process embeddings from individual CSV files and pad to max length."""
    df = pd.read_csv(INPUT_CSV)
    df = df.drop(df.columns[0], axis=1)
    
    x_dfs = [pd.read_csv(f'{merged_span_dir}/{idx}.csv') for idx in essay_ids]
    
    tokens_by_id = {}
    first_column = 'Unnamed: 0'
    if 'EMBEDDINGS_token_stride' in merged_span_dir:
        first_column = '0'
        # print(x_dfs[0].columns)
        # print(x_dfs[0].head())

    for eid, xdf in zip(essay_ids, x_dfs):
        tokens_by_id[eid] = xdf[first_column].values.tolist()

    max_len = max(len(xdf) for xdf in x_dfs)
    print(f'MAX number of tokens is: {max_len}')

    xx_dfs = []
    for xdf in tqdm(x_dfs):
        xdf = xdf.drop(columns=[first_column])
        padded = pd.concat((
            xdf,
            pd.DataFrame([[-1] * len(xdf.columns)] * (max_len - len(xdf)), columns=xdf.columns)
        ), ignore_index=True)
        xx_dfs.append(padded)

    xx = np.array([xdf.values for xdf in xx_dfs])
    yy = df[['roberta_prediction']].values.astype(int)
    
    return xx, yy, essay_ids, tokens_by_id


def split_data(xx, yy, essay_ids):
    """Split data into train and test sets."""
    return train_test_split(xx, yy, essay_ids, test_size=0.2, random_state=0)


def create_output_directory(args):
    """Create output directory based on hyperparameters."""
    prefix = args.train_set.split('_')[-1]
    directory_path = f'{DATA_DIR}/SPAN_EXPS_{prefix}_{args.epochs}_{args.batch_size}_{args.learning_rate}/'
    print(f'{directory_path=}')
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def load_or_train_model(args, xx_train, yy_train, directory_path):
    """Load existing model or train a new one."""
    model_file_path = f'{directory_path}/{MODEL_FILENAME}'
    
    if os.path.exists(model_file_path):
        # Load the model from the file
        with open(model_file_path, 'rb') as f:
            rot = pickle.load(f)
        print("Loaded the RuleOfThumb model from disk.")
    else:
        # Train new model
        print('Will initialise and fit new RoT model')
        print()
        print(f'Args passed to RuleOfThumb: y_outputs=<yy>, x_inputs=<xx>, epochs={args.epochs}, '
              f'batch_size={args.batch_size}, learning_rate={args.learning_rate}, dropout_rate={args.dropout_rate}')
        print()
        start_time = time.time()
        rot = RuleOfThumb(
            y_outputs=yy_train, 
            x_inputs=xx_train, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate, 
            dropout_rate=args.dropout_rate, 
            l1_penalty=args.l1_penalty, 
            write_file=f"{directory_path}/{HYPERPARAMS_FILENAME}"
        )
        end_time = time.time()
        with open('./TIMING.csv', 'a') as time_file:
            time_file.write(f"rot_train___{args.batch_size}_{args.learning_rate}_{args.dropout_rate},{len(yy_train)},{end_time-start_time}\n")

    
    return rot, model_file_path


def compute_and_print_stats(rot, model_file_path, performance_path, xx_train, yy_train, xx_test, yy_test, loss=None):
    """Compute and print model statistics and save to file."""
    assert rot is not None

    train_exps = []
    yy_train_pred = []

    # Process training explanations and predictions in batches
    batch_size = 128
    for i in range(0, len(xx_train), batch_size):
        batch_xx_train = xx_train[i:i + batch_size]
        batch_exps = rot.get_explanation(batch_xx_train)
        train_exps.extend(batch_exps)
        batch_pred = rot._explainer_model.predict(torch.from_numpy(batch_xx_train).to(torch.float)).detach().numpy()
        yy_train_pred.extend(batch_pred)

    yy_train_pred = np.array(yy_train_pred)

    test_exps = rot.get_explanation(xx_test)
    yy_test_pred = rot._explainer_model.predict(torch.from_numpy(xx_test).to(torch.float)).detach().numpy()

    # Open a file to write the details
    with open(performance_path, 'w') as file:
        # Write and print ROT global importance value
        rot_global_importance = f'ROT global importance value is: {rot._explainer_model.g}'
        mean_a = np.mean(rot._explainer_model.a.detach().numpy())
        mean_b = np.mean(rot._explainer_model.b.detach().numpy())
        mean_abs_a = np.mean(np.abs(rot._explainer_model.a.detach().numpy()))
        mean_abs_b = np.mean(np.abs(rot._explainer_model.b.detach().numpy()))
        mean_values = (
            f'Mean of self.a: {mean_a}, Mean of self.b: {mean_b}, '
            f'Mean of abs(self.a): {mean_abs_a}, Mean of abs(self.b): {mean_abs_b}'
        )
        
        print(rot_global_importance)
        print(mean_values)
        
        file.write(rot_global_importance + '\n')
        file.write(mean_values + '\n')
        
        if loss is not None:
            loss_string = f'Mean loss for these epochs: {loss}'
            print(loss_string)
            file.write(loss_string + '\n')

        # Calculate and write training metrics
        tn, fp, fn, tp = confusion_matrix(yy_train, yy_train_pred).ravel()
        train_metrics = (
            'TRAIN\n'
            f'TN rot=0, actual=0: {tn}\n'
            f'FP rot=1, actual=0: {fp}\n'
            f'FN rot=0, actual=1: {fn}\n'
            f'TP rot=1, actual=1: {tp}\n'
        )
        train_accuracy = accuracy_score(yy_train, yy_train_pred)
        train_accuracy_str = f'TRAIN accuracy: {train_accuracy}\n'
        print(train_metrics + train_accuracy_str)
        file.write(train_metrics + train_accuracy_str + '\n')

        # Calculate and write testing metrics
        tn, fp, fn, tp = confusion_matrix(yy_test, yy_test_pred).ravel()
        test_metrics = (
            'TEST\n'
            f'TN rot=0, actual=0: {tn}\n'
            f'FP rot=1, actual=0: {fp}\n'
            f'FN rot=0, actual=1: {fn}\n'
            f'TP rot=1, actual=1: {tp}\n'
        )
        test_accuracy = accuracy_score(yy_test, yy_test_pred)
        test_accuracy_str = f'TEST accuracy: {test_accuracy}'
        print(test_metrics + test_accuracy_str)
        file.write(test_metrics + test_accuracy_str)

        try:
            with open(model_file_path, 'wb') as f:
                pickle.dump(rot, f)
            print(f"Saved the {rot} to {model_file_path}.")
        except:
            print(f"Could not save the {rot} to {model_file_path}.")


def generate_predictions_and_explanations(rot, xx_train, yy_train, xx_test, yy_test, essay_ids_train, essay_ids_test):
    """Generate predictions and explanations for train and test sets."""
    # Test predictions and explanations
    start_time = time.time()
    test_exps = rot.get_explanation(xx_test)
    end_time = time.time()
    with open('./TIMING.csv', 'a') as time_file:
        time_file.write(f"rot_explanation,{len(test_exps)},{end_time-start_time}\n")

    yy_test_pred = rot._explainer_model.predict(torch.from_numpy(xx_test).to(torch.float)).detach().numpy()
    predictions_test_df = pd.DataFrame({'index': essay_ids_test, 'rot_prediction': yy_test_pred.flatten()})
    explanations_test_dict = {essay_id: exp for essay_id, exp in zip(essay_ids_test, test_exps)}

    # Train predictions and explanations
    train_exps = []
    yy_train_pred = []

    # Process training explanations and predictions in batches
    batch_size = 128
    for i in range(0, len(xx_train), batch_size):
        batch_xx_train = xx_train[i:i + batch_size]
        batch_exps = rot.get_explanation(batch_xx_train)
        train_exps.extend(batch_exps)
        batch_pred = rot._explainer_model.predict(torch.from_numpy(batch_xx_train).to(torch.float)).detach().numpy()
        yy_train_pred.extend(batch_pred)

    yy_train_pred = np.array(yy_train_pred)
    predictions_train_df = pd.DataFrame({'index': essay_ids_train, 'rot_prediction': yy_train_pred.flatten()})
    explanations_train_dict = {essay_id: exp for essay_id, exp in zip(essay_ids_train, train_exps)}

    return (predictions_train_df, explanations_train_dict, 
            predictions_test_df, explanations_test_dict)


def select_dataset_for_output(args, predictions_train_df, explanations_train_dict, 
                            predictions_test_df, explanations_test_dict, directory_path):
    """Select which dataset to use for output based on use_test flag."""
    use_test = args.use_test
    print(f'{use_test=}')
    print(f'{type(use_test)=}')
    
    predictions_df = predictions_train_df
    explanations_dict = explanations_train_dict
    dataset_label = 'TRAIN'
    
    if use_test:
        predictions_df = predictions_test_df
        explanations_dict = explanations_test_dict
        dataset_label = 'TEST'
    
    with open(f'{directory_path}/{DATASET_VISUALISED_FILENAME}', 'w') as f:
        f.write(f'{dataset_label}\n')
    
    return predictions_df, explanations_dict


def create_and_save_meta_csv(predictions_df, directory_path):
    """Create and save the meta CSV with merged predictions."""
    score_df = pd.read_csv(SCORE_CSV)
    score_df = score_df.drop(['Unnamed: 0'], axis=1)
    print(score_df.head())
    
    score_df = score_df.merge(predictions_df, on='index', how='inner')
    score_df.to_csv(f'{directory_path}/{META_CSV_FILENAME}')
    
    return score_df



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



def save_explanations_by_category(explanations_dict, tokens_by_id, score_df, directory_path):
    """Save explanation files organized by prediction categories."""
    for essay_id, explanation in explanations_dict.items():
        explanation_df = pd.DataFrame(index=tokens_by_id[essay_id])
        explanation_df['importance'] = explanation[:len(explanation_df)]
        # explanation_df['scaled_importance'] = scale_importances(explanation_df['importance'], use_pow=5)
        explanation_df['scaled_importance'] = scale_importances(explanation_df['importance'])

        # Determine the subdirectory based on the combination of roberta_prediction, human_label, and rot_prediction
        row = score_df[score_df['index'] == essay_id].iloc[0]
        subdirectory = f"{row['roberta_prediction']}_{row['ground_truth_judgement']}_{row['rot_prediction']}"

        # Create the directory if it doesn't exist
        sub_directory_path = f'{directory_path}/{subdirectory}'
        os.makedirs(sub_directory_path, exist_ok=True)

        # Save the explanation DataFrame to the appropriate subdirectory
        explanation_df.to_csv(f'{sub_directory_path}/{essay_id}.csv')


def main():
    """Main function to orchestrate the ROT model training pipeline."""
    # Parse arguments
    args = parse_arguments()

    # # Setup constants
    # initialise_constants(args)
    
    # Create output directory
    directory_path = create_output_directory(args)

    start_time_ = time.time()

    # Load initial data
    df, all_essay_ids = load_initial_data()
    
    # Load or process embeddings
    xx_PredEx, yy_PredEx, essay_ids_PredEx, tokens_by_id_PredEx = load_or_process_embeddings(all_essay_ids, TEST_MERGED_SPAN_DIR)
    xx_fixed, yy_fixed, essay_ids_fixed, tokens_by_id_fixed = load_or_process_embeddings(all_essay_ids, f'./DATA/{args.train_set}')

    for eid in tokens_by_id_PredEx:
        assert eid in tokens_by_id_fixed
    for eid in tokens_by_id_fixed:
        assert eid in tokens_by_id_PredEx
    
    # Split data - Normal Fit mode:
    xx_train, _, yy_train, _, essay_ids_train, essay_ids_test_1 = split_data(xx_fixed, yy_fixed, all_essay_ids)
    # xx_test, _, yy_test, _, essay_ids_test, _ = split_data(xx_PredEx, yy_PredEx, all_essay_ids)
    _, xx_test, _, yy_test, essay_ids_train_2, essay_ids_test = split_data(xx_PredEx, yy_PredEx, all_essay_ids)

    for eid1, eid2 in zip(essay_ids_train, essay_ids_train_2):
        assert eid1 == eid2
    for eid1, eid2 in zip(essay_ids_test_1, essay_ids_test):
        assert eid1 == eid2

    # # Split data - OverFit mode:
    # xx_train, yy_train, essay_ids_train = xx_fixed, yy_fixed, all_essay_ids
    # _, xx_test, _, yy_test, _, essay_ids_test = split_data(xx_PredEx, yy_PredEx, all_essay_ids)
    
    # # Split data - Super OverFit mode:
    # xx_train = xx_test
    # yy_train = yy_test
    # essay_ids_train = essay_ids_test
    
    # Load or train model
    rot, model_file_path = load_or_train_model(args, xx_train, yy_train, directory_path)
    
    # Compute and print statistics
    compute_and_print_stats(rot, model_file_path, f'{directory_path}/{PERFORMANCE_FILENAME}', 
                          xx_train, yy_train, xx_test, yy_test)
    
    # Generate predictions and explanations
    (predictions_train_df, explanations_train_dict, 
     predictions_test_df, explanations_test_dict) = generate_predictions_and_explanations(
        rot, xx_train, yy_train, xx_test, yy_test, essay_ids_train, essay_ids_test)
    
    # Select dataset for output
    predictions_df, explanations_dict = select_dataset_for_output(
        args, predictions_train_df, explanations_train_dict, 
        predictions_test_df, explanations_test_dict, directory_path)
    
    # Create and save meta CSV
    score_df = create_and_save_meta_csv(predictions_df, directory_path)
    
    # Save explanations by category
    save_explanations_by_category(explanations_dict, tokens_by_id_PredEx, score_df, directory_path)

    end_time_ = time.time()

    with open('./TIMING.csv', 'a') as time_file:
        time_file.write(f"rot_combined,FULL_SCRIPT,{end_time_-start_time_}\n")


if __name__ == "__main__":
    main()


