import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser(description='Result Analysis')
parser.add_argument('--directory', type=str, required=True, help='directory with explainer outputs')
parser.add_argument('--plots', type=str, required=True, help='directory to save plots')
parser.add_argument('--metadata', type=str, required=True, help='csv file with labels and predictions')
parser.add_argument('--mask', type=int, required=False, default=0, help='ignore all spans with length lesser than this number')
parser.add_argument('--flip', action='store_true', help='flip all signs')
args = parser.parse_args()

df = pd.read_csv(args.metadata)
# print(df.head())


# For plotting
all_fpr = []
all_tpr = []
all_precision = []
all_recall = []
all_weights_ = []

all_gt = []
all_importances = []
all_weights = []


prauc = []
auroc = []
wprauc = []
wauroc = []
rprauc = []
rauroc = []
minweights = []
avgweights = []
for review_id, target in zip(df['review_id'], df['rot_target']):
    if target not in [0,1]:
        continue
    fpth = f'{args.directory}/{review_id}.csv'
    explanation_df = pd.read_csv(fpth)
    spans = explanation_df['Unnamed: 0'].values.tolist()
    spans = [str(s) for s in spans]
    importances = explanation_df['importance'].values.tolist()
    gt = [1 if span.startswith('___') else 0 for span in spans]
    if args.flip:
        gt = [1-g for g in gt]
        importances = [-i for i in importances]
    weights = [len(span)-3 if span.startswith('___') else len(span) for span in spans]
    if min(weights) < args.mask:
        # pass
        continue
    minweights.append(min(weights))
    avgweights.append(sum(weights)/len(weights))
    if target < 0.5:
        importances = [-imp for imp in importances]
    auroc.append(roc_auc_score(gt, importances))
    prauc.append(average_precision_score(gt, importances))
    wauroc.append(roc_auc_score(gt, importances, sample_weight=weights))
    wprauc.append(average_precision_score(gt, importances, sample_weight=weights))
    random_imps = np.random.random(len(importances))
    rauroc.append(roc_auc_score(gt, random_imps, sample_weight=weights))
    rprauc.append(average_precision_score(gt, random_imps, sample_weight=weights))
    fpr, tpr, _ = roc_curve(gt, importances)
    precision, recall, _ = precision_recall_curve(gt, importances)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_precision.append(precision)
    all_recall.append(recall)

    all_weights.extend(weights)
    all_weights_.append([min(weights)] * len(fpr))
    all_gt.extend(gt)
    all_importances.extend(importances)

print(f'total number of reviews: {len(minweights)}')
# print(f'average prauc: {sum(prauc)/len(prauc)}')
# print(f'average auroc: {sum(auroc)/len(auroc)}')
print(f'average wprauc: {sum(wprauc)/len(wprauc)}')
print(f'average wauroc: {sum(wauroc)/len(wauroc)}')
print(f'random average wprauc: {sum(rprauc)/len(wprauc)}')
print(f'random average wauroc: {sum(rauroc)/len(wauroc)}')

# print(f'{np.histogram(minweights)=}')
# print(f'{np.histogram(avgweights)=}')

# combined_tpr = []
# combined_fpr = []
# combined_roc_weights = []

# Plot ROC curves
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
count = 0
for fpr, tpr, wt in zip(all_fpr, all_tpr, all_weights_):
    # combined_tpr.extend(tpr)
    # combined_fpr.extend(fpr)
    # combined_roc_weights.extend(wt)
    count += 1
    # print(fpr)
    # print(tpr)

    # plt.plot(fpr, tpr, )
    # plt.scatter(fpr, tpr, color='darkgreen', alpha=0.05, marker='.')
    plt.scatter(fpr, tpr, c=wt, cmap='viridis', alpha=0.05, marker='.')    
    # if count == 1:
    #     plt.plot(fpr, tpr, color='darkgreen', alpha=0.05, linewidth=0.5, label='case')
    # else:
    #     plt.plot(fpr, tpr, color='darkgreen', alpha=0.05, linewidth=0.5)

    # plt.step(fpr, tpr, where='pre', alpha=0.2, linewidth=0.5, color='C0', label='pre')
    # plt.step(fpr, tpr, where='post', alpha=0.2, linewidth=0.5, color='C1', label='post')
    # plt.step(fpr, tpr, where='mid', alpha=0.2, linewidth=0.5, color='C3', label='mid')
    # plt.legend()
    # plt.show()
    if count > 5:
        pass
        # break
# plt.plot([0,1],[0,1],'k--', label='Random')
fpr, tpr, _ = roc_curve(all_gt, all_importances)
plt.plot(fpr, tpr, color='darkgreen', linestyle='-.', label='overall')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig(f'{args.plots}/ROC.png')
plt.savefig(f'{args.plots}/ROC.pdf', dpi=300)
# plt.show()


# combined_precision = []
# combined_recall = []
# combined_pr_weights = []

# Plot PR curves
count = 0
for precision, recall, wt in zip(all_precision, all_recall, all_weights):
    # combined_precision.extend(precision)
    # combined_recall.extend(recall)
    # combined_pr_weights.extend(wt)
    count += 1
    # precision = np.insert(precision, 0, 1.0)
    # recall = np.insert(recall, 0, 0.0)
    # print(precision)
    # print(recall)
    # plt.step(recall, precision, color='fuchsia', where='post', alpha=0.2, linewidth=0.5)
    if count == 1:
        plt.plot(recall, precision, color='fuchsia', alpha=0.05, linewidth=0.5, label='case')
    else:
        plt.plot(recall, precision, color='fuchsia', alpha=0.05, linewidth=0.5)

    # plt.step(recall, precision, where='pre', alpha=0.2, linewidth=0.5, color='C0', label='pre')
    # plt.step(recall, precision, where='post', alpha=0.2, linewidth=0.5, color='C1', label='post')
    # plt.step(recall, precision, where='mid', alpha=0.2, linewidth=0.5, color='C3', label='mid')
    # plt.legend()
    # plt.show()
    if count > 5:
        pass
        # break
pr, rc, _ = precision_recall_curve(all_gt, all_importances)
plt.plot(rc, pr, color='fuchsia', linestyle='-.', label='PR')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.tight_layout()
plt.savefig(f'{args.plots}/PR.png')
plt.savefig(f'{args.plots}/PR.pdf', dpi=300)
# plt.show()


# print(f'COMBINED ROC-AUC: {roc_auc_score(all_gt, all_importances)}')
# print(f'COMBINED PR-AUC: {average_precision_score(all_gt, all_importances)}')
# print(f'COMBINED wROC-AUC: {roc_auc_score(all_gt, all_importances, sample_weight=all_weights)}')
# print(f'COMBINED wPR-AUC: {average_precision_score(all_gt, all_importances, sample_weight=all_weights)}')
fpr, tpr, _ = roc_curve(all_gt, all_importances)
pr, rc, _ = precision_recall_curve(all_gt, all_importances)
plt.plot(fpr, tpr, color='darkgreen', linestyle='-.', label='ROC')
plt.plot(rc, pr, color='fuchsia', linestyle='-.', label='PR')
plt.legend()
plt.tight_layout()
plt.savefig(f'{args.plots}/combined.png')
plt.savefig(f'{args.plots}/combined.pdf', dpi=300)
# plt.show()


print()
print()
print()
