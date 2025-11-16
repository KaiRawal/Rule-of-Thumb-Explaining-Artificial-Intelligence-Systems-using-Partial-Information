import pandas as pd
import numpy as np
import torch
import pickle
import os

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rule_of_thumb import RuleOfThumb

import argparse
import sys

import copy

parser = argparse.ArgumentParser(description='Training Rule of Thumb model')
parser.add_argument('--epochs', type=int, default=128, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.000002, help='Learning rate')
# parser.add_argument('--learning_rate', type=float, default=0.000002, help='Learning rate')
parser.add_argument('--train_file', type=str, required=True, help='training metadata csv')
parser.add_argument('--test_file', type=str, required=True, help='test metadata csv')
parser.add_argument('--explanation_dir', type=str, required=True, help='output directory')
parser.add_argument('--cache_dir', type=str, required=True, help='cache directory')

# parser.add_argument('--help', action='help', help='Show this help message and exit')
args = parser.parse_args()

df_train = pd.read_csv(args.train_file)
df_train = df_train.drop(columns=['Unnamed: 0'])
df_test = pd.read_csv(args.test_file)
df_test = df_test.drop(columns=['Unnamed: 0'])
# print(df_train.tail())
# print(df_test.head())

tokens_by_id = {}
xx_train = None
yy_train = None
xx_test = None
yy_test = None

yy_human = df_test[['human_prediction']].values.astype(int)
yy_gpt = df_test[['gpt_prediction']].values.astype(int)


if os.path.exists(f'{args.cache_dir}/id2tok.pkl'):
    print('Loading CACHED data ...')
    xx_train = np.load(f'{args.cache_dir}/xtrain.npy')
    xx_test = np.load(f'{args.cache_dir}/xtest.npy')
    yy_train = np.load(f'{args.cache_dir}/ytrain.npy')
    yy_test = np.load(f'{args.cache_dir}/ytest.npy')
    with open(f'{args.cache_dir}/id2tok.pkl', 'rb') as f:
        tokens_by_id = pickle.load(f)
else:
    print('PREPROCESSING DATA ...')
    x_train_dfs = [pd.read_csv(fpth) for fpth in df_train['token_embedding_file']]
    x_test_dfs = [pd.read_csv(fpth) for fpth in df_test['merged_embedding_file']]
    print('\t1.1 loaded to memory ...')


    max_len = -1
    xx_train_dfs = []
    for rid, xdf in zip(df_train['token_embedding_file'], x_train_dfs):
        review_id = rid.split('/')[-1][:-4]
        tokens_by_id[review_id] = xdf['Unnamed: 0'].values.tolist()
        xdf = xdf.drop(columns=['Unnamed: 0'])
        if len(xdf) > max_len:
            max_len = len(xdf)
        xx_train_dfs.append(xdf)
    print(f'MAX number of tokens is: {max_len}')
    print(f'\t1.2 traversed x_train_dfs ... {len(x_train_dfs)=}')
    x_train_dfs = None


    xx_train = np.array([
        pd.concat(
            [xdf] + (
                [pd.DataFrame([[-1] * len(xdf.columns)] * (max_len - len(xdf)), columns=xdf.columns)]
                if max_len > len(xdf) else []
            ),
            ignore_index=True
        ).values
        for xdf in xx_train_dfs
    ])
    print(f'\t1.3 created x_train array ... {xx_train.shape=}')
    xx_train_dfs = None


    max_len = -1
    xx_test_dfs = []
    for rid, xdf in zip(df_test['merged_embedding_file'], x_test_dfs):
        review_id = rid.split('/')[-1][:-4]
        tokens_by_id[review_id] = xdf['Unnamed: 0'].values.tolist()
        xdf = xdf.drop(columns=['Unnamed: 0'])
        if len(xdf) > max_len:
            max_len = len(xdf)
        xx_test_dfs.append(xdf)
    print(f'MAX number of spans is: {max_len}')
    print(f'\t1.4 traversed x_test_dfs ... {len(x_test_dfs)=}')
    x_test_dfs = None


    xx_test = np.array([
        pd.concat(
            [xdf] + (
                [pd.DataFrame([[-1] * len(xdf.columns)] * (max_len - len(xdf)), columns=xdf.columns)]
                if max_len > len(xdf) else []
            ),
            ignore_index=True
        ).values
        for xdf in xx_test_dfs
    ])
    print(f'\t1.5 created x_test array ... {xx_test.shape=}')
    xx_test_dfs = None


    yy_train = df_train[['rot_target']].values.astype(int)
    yy_test = df_test[['rot_target']].values.astype(int)
    print(f'{xx_train.shape=}')
    print(f'{xx_test.shape=}')
    print(f'{yy_train.shape=}')
    print(f'{yy_test.shape=}')
    print('\t1.5 final save to cache')
    np.save(f'{args.cache_dir}/xtrain.npy', xx_train)
    np.save(f'{args.cache_dir}/xtest.npy', xx_test)
    np.save(f'{args.cache_dir}/ytrain.npy', yy_train)
    np.save(f'{args.cache_dir}/ytest.npy', yy_test)
    with open(f'{args.cache_dir}/id2tok.pkl', 'wb') as f:
        pickle.dump(tokens_by_id, f)


# xx_train, xx_test, yy_train, yy_test, essay_ids_train, essay_ids_test = train_test_split(xx, yy, essay_ids, test_size=0.3, random_state=0)

print()
print(f'Args passed to RuleOfThumb: y_outputs=<yy>, x_inputs=<xx>, epochs={args.epochs}, batch_size={args.batch_size}, learning_rate={args.learning_rate}, dropout_rate=0.5')
print()

rot = None

# Check if the model file exists
model_file_path = f'{args.cache_dir}/rot_{args.batch_size}_{args.epochs}_{args.learning_rate}_model.pkl'
if os.path.exists(model_file_path):
    # Load the model from the file
    with open(model_file_path, 'rb') as f:
        rot = pickle.load(f)
    print("Loaded the RuleOfThumb model from disk.")
else:
    # Pickle the trained PyTorch model
    rot = RuleOfThumb(y_outputs=yy_train, x_inputs=xx_train, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, dropout_rate=0.5)
    with open(model_file_path, 'wb') as f:
        pickle.dump(rot, f)
    print("Saved the RuleOfThumb model to disk.")

assert rot is not None

# train_exps = rot.get_explanation(xx_train)
# print(f'{train_exps.shape}')
# print([f'{texp.shape}' for texp in train_exps])
# for texp in train_exps[:5]:
#     print(list(texp))
print('\t2.1 getting train predictions ...')

# yy_train_pred = rot._explainer_model.predict(torch.from_numpy(xx_train).to(torch.float)).detach().numpy()
# yy_train_pred = yy_train
# xx_train = None

# Batch the predictions to avoid holding the entire xx_train in memory at once
batch_size = 128
n_train = xx_train.shape[0]
train_preds = []
with torch.no_grad():
    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        batch = torch.from_numpy(xx_train[start:end]).to(torch.float)
        batch_pred = rot._explainer_model.predict(batch).detach().numpy()
        train_preds.append(batch_pred.reshape(batch_pred.shape[0],))

yy_train_pred = np.concatenate(train_preds, axis=0)
xx_train = None


print('\t2.2 getting test explanations ...')
test_exps = rot.get_explanation(xx_test)
print('\t2.3 getting test predictions ...')
yy_test_pred = rot._explainer_model.predict(torch.from_numpy(xx_test).to(torch.float)).detach().numpy()

# print(xx_train.shape)
# print(xx_test.shape)
# print(yy_train_pred.shape)
# print(yy_test_pred.shape)
# print(yy_test.shape)
# print(yy_train.shape)

print(f'ROT global importance value is: ' + str(rot._explainer_model.g))

tn, fp, fn, tp = confusion_matrix(yy_train, yy_train_pred).ravel()
print('TRAIN')
print(f'TN rot=0, actual=0: {tn}')
print(f'FP rot=1, actual=0: {fp}')
print(f'FN rot=0, actual=1: {fn}')
print(f'TP rot=1, actual=1: {tp}')
train_accuracy = accuracy_score(yy_train, yy_train_pred)
print(f'TRAIN accuracy: {train_accuracy}')
print()


tn, fp, fn, tp = confusion_matrix(yy_test, yy_test_pred).ravel()
print('TEST')
print(f'TN rot=0, actual=0: {tn}')
print(f'FP rot=1, actual=0: {fp}')
print(f'FN rot=0, actual=1: {fn}')
print(f'TP rot=1, actual=1: {tp}')
test_accuracy = accuracy_score(yy_test, yy_test_pred)
print(f'TEST accuracy: {test_accuracy}')




predictions_test_df = pd.DataFrame({'review_id': df_test['review_id'], 'rot_prediction': yy_test_pred.flatten(), 'rot_target': df_test['rot_target'].values.flatten(), 'human_review_label': yy_human.flatten(), 'gpt_review_label': yy_gpt.flatten()})
explanations_test_dict = {rid: exp for rid, exp in zip(df_test['review_id'], test_exps)}
# print(f'{explanations_test_dict=}')

# predictions_train_df = pd.DataFrame({'review_id': df_train['review_id'], 'rot_prediction': yy_train_pred.flatten(), 'rot_target': df_train['rot_target']})
# explanations_train_dict = {rid: exp for rid, exp in zip(df_train['review_id'], train_exps)}


predictions_test_df.to_csv(f'{args.cache_dir}/meta.csv')


# def scale_importances(importances):
#     pos_importances = importances[importances > 0]
#     neg_importances = importances[importances < 0]
#     maxp = np.max(pos_importances)
#     minn = np.min(pos_importances)
#     absmax = None
#     if maxp + minn > 0:
#         absmax = maxp
#     else:
#         absmax = -minn
#     return importances * 100 / absmax
def scale_importances(imps, use_logarithm=False, use_pow=-1):
    """Scale importance values for visualization."""
    imps = np.array(imps)
    # p1 = np.hstack((imps[-1], imps[:-1]))
    # n1 = np.hstack((imps[1:], imps[0]))
    # importances = 2*imps + p1 + n1
    importances = imps
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

for rid, explanation in tqdm(explanations_test_dict.items()):
    explanation_df = pd.DataFrame(index=tokens_by_id[rid])
    explanation_df['importance'] = explanation[:len(explanation_df)]
    explanation_df['scaled_importance'] = scale_importances(explanation_df['importance'])
    explanation_df.to_csv(f'{args.explanation_dir}/{rid}.csv')
    # print(f'{args.explanation_dir}/{rid}.csv')



