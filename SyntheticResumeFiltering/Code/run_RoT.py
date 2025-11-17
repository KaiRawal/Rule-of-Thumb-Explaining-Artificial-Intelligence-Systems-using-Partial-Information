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

from rule_of_thumb import RuleOfThumb

import argparse
import sys

import copy

parser = argparse.ArgumentParser(description='Training Rule of Thumb model')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=5000, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
# parser.add_argument('--help', action='help', help='Show this help message and exit')
args = parser.parse_args()

df_all = pd.read_csv('./DATA/rot_input_embeddings.csv')
df_all = df_all.drop(df_all.columns[0], axis=1)

df = df_all
# pos_df = df_all[df_all['llm_prediction'] == 1]
# neg_df = df_all[df_all['llm_prediction'] == 0]
# neg_sample = neg_df.sample(n=len(pos_df), random_state=0)
# df = pd.concat([pos_df.copy(), neg_sample.copy()])




# xx = df[df.columns[1:-1]].values.astype(float)
essay_ids = list(df['index'])

xx = []


x_dfs = [pd.read_csv(f'./DATA/EMBEDDINGS/{idx}.csv') for idx in essay_ids]
tokens_by_id = {}

for eid, xdf in zip(essay_ids, x_dfs):
    tokens_by_id[eid] = xdf['Unnamed: 0'].values.tolist()

max_len = -1
xx_dfs = []

# for xdf, race, gender in zip(x_dfs, list(df['race']), list(df['gender'])):
for xdf in x_dfs:
    xdf = xdf.drop(columns=['Unnamed: 0'])
    # xdf['race'] = race
    # xdf['gender'] = gender
    if len(xdf) > max_len:
        max_len = len(xdf)
    xx_dfs.append(xdf)
    # print(xdf.values.shape)

print(f'MAX number of tokens is: {max_len}')

# xx_dfs = [xdf.append(pd.DataFrame([[-1] * len(xdf.columns)] * (max_len - len(xdf)), columns=xdf.columns), ignore_index=True) for xdf in xx_dfs]
xx_dfs = [
    pd.concat(
        [xdf] + (
            [pd.DataFrame([[-1] * len(xdf.columns)] * (max_len - len(xdf)), columns=xdf.columns)]
            if max_len > len(xdf) else []
        ),
        ignore_index=True
    )
    for xdf in xx_dfs
]
# xx_dfs = [
#     pd.concat(
#         [xdf, pd.DataFrame([[-1] * len(xdf.columns)] * (max_len - len(xdf)), columns=xdf.columns)],
#         ignore_index=True
#     )
#     for xdf in xx_dfs
# ]


# for xdf in xx_dfs:
#     # print(xdf.columns)
#     # mean_row = xdf.mean(numeric_only=False)
#     pad_row = np.array([-1] * len(xdf.columns)) # xdf.mean(numeric_only=False)
#     while len(xdf) < max_len:
#         xdf.loc[len(xdf)] = pad_row
#     # print(xdf.values.shape)


xx = [xdf.values for xdf in xx_dfs]

xx = np.array(xx)

yy = df[['llm_prediction']].values.astype(int)

print(f'{sum(yy)=} / {len(yy)=}')

xx_train, xx_test, yy_train, yy_test, essay_ids_train, essay_ids_test = train_test_split(xx, yy, essay_ids, test_size=0.3, random_state=0)


print()
print()
print()
print()

print(f'Args passed to RuleOfThumb: y_outputs=<yy>, x_inputs=<xx>, epochs={args.epochs}, batch_size={args.batch_size}, learning_rate={args.learning_rate}, dropout_rate=0.5')
print()

rot = None

# Check if the model file exists
model_file_path = './DATA/rot_model.pkl'
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

train_exps = rot.get_explanation(xx_train)
# print(f'{train_exps.shape}')
# print([f'{texp.shape}' for texp in train_exps])
# for texp in train_exps[:5]:
#     print(list(texp))
yy_train_pred = rot._explainer_model.predict(torch.from_numpy(xx_train).to(torch.float)).detach().numpy()
test_exps = rot.get_explanation(xx_test)
yy_test_pred = rot._explainer_model.predict(torch.from_numpy(xx_test).to(torch.float)).detach().numpy()


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




score_df = pd.read_csv('DATA/rot_input_embeddings.csv')
score_df = score_df.drop(['Unnamed: 0'], axis=1)
print(score_df.head())

predictions_test_df = pd.DataFrame({'index': essay_ids_test, 'rot_prediction': yy_test_pred.flatten()})
explanations_test_dict = {essay_id: exp for essay_id, exp in zip(essay_ids_test, test_exps)}
predictions_train_df = pd.DataFrame({'index': essay_ids_train, 'rot_prediction': yy_train_pred.flatten()})
explanations_train_dict = {essay_id: exp for essay_id, exp in zip(essay_ids_train, train_exps)}

# predictions_df = predictions_train_df
# explanations_dict = explanations_train_dict
predictions_df = predictions_test_df
explanations_dict = explanations_test_dict

score_df = score_df.merge(predictions_df, on='index', how='inner')

score_df.to_csv('./DATA/meta.csv')


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

for essay_id, explanation in explanations_dict.items():
    # print(f'{essay_id=}')
    # print(f'{explanation.shape=}')
    # print(f'{type(explanation)=}')
    explanation_df = pd.DataFrame(index=tokens_by_id[essay_id])
    explanation_df['importance'] = explanation[:len(explanation_df)]
    explanation_df['scaled_importance'] = scale_importances(explanation_df['importance'])
    explanation_df.to_csv(f'./DATA/TOKEN_EXPS/{essay_id}.csv')



