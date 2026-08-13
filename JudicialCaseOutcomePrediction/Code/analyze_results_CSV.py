import os
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib.pyplot as plt
import numpy as np


USE_ABS = False
USE_RELU = True

# fixed
USE_ROBERTA = None
FLIP_0 = False
FLIP_1 = False

FORMULA = None
METRIC = None

def get_metric_name():
    return 'PRAUC' if METRIC == 'pr_auc' else 'AUC'

def sc_score(y_true, y_score, sample_weight=None):
    if METRIC == 'pr_auc':
        return average_precision_score(y_true, y_score, sample_weight=sample_weight)
    return roc_auc_score(y_true, y_score, sample_weight=sample_weight)

class Case:
    def __init__(self):
        self.annotations = []       # 1 if span is annotated (has ___), 0 otherwise
        self.rot_preds = []         # RoT importance score
        self.lengths = []           # character length of each span
        self.case_label = None      # judgement
        self.roberta_prediction = None
        self.rot_prediction = None
        self.case_name = ''         # filename without .csv
        self.is_not_rot = False
    
    def _display(self, uids=[1506, 895, 2315, 252, 721, 317]):
        return False
        for uid in uids:
            if self.case_name.startswith(f'{uid}__'):
                return True
        return False

    def get_human_labels(self, use_roberta=True, flip_0=False, flip_1=False):
        return self.annotations

        flip_test = None
        if use_roberta:
            flip_test = int(self.roberta_prediction)
        else:
            flip_test = int(self.case_label)
        if (flip_1 and flip_test == 1) or (flip_0 and flip_test == 0):
            return [abs(x-1) for x in self.annotations]
        if self._display():
            print(f'PredEx \t {self.case_name} \t {self.annotations}')
        return self.annotations

    def get_rot_labels(self, use_abs=USE_ABS, use_relu=USE_RELU, legacy=False):
        result = self.rot_preds
        if FORMULA is None:
            assert False
        elif FORMULA == 'linear':
            result = [(r+1)/2 for r in result]
        elif FORMULA == 'positived':
            result = [1+r if r < 0 else r for r in result]
        elif FORMULA == 'conditional':
            if self.roberta_prediction > 0:
                result = [r if r > 0 else 0 for r in result]
            else:
                result = [1+r if r < 0 else 0 for r in result]
        elif FORMULA == 'flip':
            if self.roberta_prediction > 0:
                # result = [r if r > 0 else 0 for r in result]
                result = [r for r in result]
            else:
                # result = [-r if r < 0 else 0 for r in result]
                result = [-r for r in result]
        elif FORMULA == 'flip_clip':
            if self.roberta_prediction > 0:
                result = [r if r > 0 else 0 for r in result]
                # result = [r for r in result]
            else:
                result = [-r if r < 0 else 0 for r in result]
                # result = [-r for r in result]
        else:
            assert False
        return result

        result = self.rot_preds
        if use_abs:
            result = [abs(r) for r in result]
        elif use_relu:
            if self.roberta_prediction > 0:
                result = [r if r > 0 else 0 for r in result]
            else:
                result = [1-abs(r) if r < 0 else 0 for r in result]
        elif legacy:
            if self.is_not_rot:
                if self.roberta_prediction > 0:
                    result = [r if r > 0 else 0 for r in result]
                else:
                    result = [-r if r < 0 else 0 for r in result]
            else:
                if self.rot_prediction > 0:
                    result = [r if r > 0 else 0 for r in result]
                else:
                    result = [-r if r < 0 else 0 for r in result]
        else:
            result = [0.5 + r/2 for r in result]
        if self._display():
            print(f'RoT probs \t {self.case_name} \t {result}')
        return result
    
    def compute_auc(self):
        sample_weights = self.get_span_lengths()
        y_preds = self.get_rot_labels()
        y_actual = self.get_human_labels()
        try:
            auc = sc_score(y_actual, y_preds)
            w_auc = sc_score(y_actual, y_preds, sample_weight=sample_weights)
        except ValueError:
            pass
            # print(f"{get_metric_name()}: Cannot be computed (possibly due to single class in y_true).")
            # print(f'{self.case_name=}')
            # print(f'{y_actual=}')
            # print(f'{y_preds=}')
            # print()
            raise ValueError(f'{get_metric_name()} computations failed for {self.case_name}: {y_preds=} {y_actual=}')
        if self._display():
            print(f'{get_metric_name()} \t {self.case_name} \t {auc}')
            print(f'w{get_metric_name()} \t {self.case_name} \t {w_auc}')
        return auc, w_auc
    
    def compute_accuracy(self, threshold=0.5):
        sample_weights = self.get_span_lengths()
        y_preds = self.get_rot_labels()
        y_actual = self.get_human_labels()
        binary_preds = [1 if p > threshold else 0 for p in y_preds]
        acc = accuracy_score(y_actual, binary_preds)
        w_acc = accuracy_score(y_actual, binary_preds, sample_weight=sample_weights)
        if self._display():
            print(f'ACCuracy \t {self.case_name} \t {acc}')
            print(f'w_ACCuracy \t {self.case_name} \t {w_acc}')
        return acc, w_acc

    def compute_best_accuracy(self, weighted=False):
        rot_probs = self.get_rot_labels()
        h = 0.000000001
        thresholds = [prob - h for prob in rot_probs] + [prob + h for prob in rot_probs]
        all_accs = []
        for thresh in thresholds:
            acc, w_acc = self.compute_accuracy(threshold=thresh)
            if weighted:
                all_accs.append(w_acc)
            else:
                all_accs.append(acc)
        if self._display():
            print(f'best accuracy \t {self.case_name} \t {weighted=} \t {max(all_accs)} \t-\t {all_accs}')
        return max(all_accs)

    def get_span_lengths(self):
        return self.lengths
    
    def compute_mpt(self):
        explainer_imps = self.get_rot_labels()
        ground_truths = self.get_human_labels()
        sample_weights = self.get_span_lengths()
        num = 0
        wnum = 0
        den = 0
        wden = 0
        for exp, gt, wt in zip(explainer_imps, ground_truths, sample_weights):
            if exp < 0:
                exp = 0
            if gt > 0.5:
                num += exp
                wnum += wt * exp
            else:
                if exp > 0:
                    den += exp
                    wden += wt * exp
        # return num/den, wnum/wden # explodes to infinity if den is 0
        res = 0
        wres = 0
        if num+den > 0:
            res = num / (num+den)
        if wnum+wden > 0:
            wres = wnum / (wnum+wden)
        return res, wres # range is from 0 to 1 like other metrics
        

# def show(cases):
#     all_valid_cases = {}
#     for case in cases:
#         if case.case_label == case.roberta_prediction:
#             all_valid_cases[case.case_name] = min(case.get_span_lengths())
#     for case_name in all_valid_cases:
#         if all_valid_cases[case_name] >= 70:
#             print(f'{case_name}.csv')
#     return
#     sorted_cases = dict(sorted(all_valid_cases.items(), key=lambda item: item[1], reverse=True))
#     for iindex, case_name in enumerate(sorted_cases):
#         print(f'{iindex} \t {case_name} : {sorted_cases[case_name]}')


def experiment(cases, subset_roberta_match_predex=False, verbose=True, index=" ", comment=" ", formula=' '):
    y_preds = []
    y_actual = []
    sample_weights = []
    
    prec_aucs = []
    prec_accs = []
    w_prec_aucs = []
    w_prec_accs = []
    best_accuracies = []
    best_weighted_accuracies = []
    mpts = []
    w_mpts = []
    
    case_count = 0
    for case in cases:
        if subset_roberta_match_predex:
            if case.roberta_prediction != case.case_label:
                continue
        u_auc, w_auc = case.compute_auc()
        u_acc, w_acc = case.compute_accuracy()
        b_acc = case.compute_best_accuracy(weighted=False)
        b_w_acc = case.compute_best_accuracy(weighted=True)
        prec_aucs.append(u_auc)
        w_prec_aucs.append(w_auc)
        prec_accs.append(u_acc)
        w_prec_accs.append(w_acc)
        best_accuracies.append(b_acc)
        best_weighted_accuracies.append(b_w_acc)

        mpt, wmpt = case.compute_mpt()
        mpts.append(mpt)
        w_mpts.append(wmpt)
        case_count += 1
    
    # Compute statistics
    unweighted_mean_auc = sum(prec_aucs)/len(prec_aucs)
    unweighted_mean_accuracy = sum(prec_accs)/len(prec_accs)
    weighted_mean_auc = sum(w_prec_aucs)/len(w_prec_aucs)
    weighted_mean_accuracy = sum(w_prec_accs)/len(w_prec_accs)
    best_mean_accuracy = sum(best_accuracies)/len(best_accuracies)
    best_weighted_mean_accuracy = sum(best_weighted_accuracies)/len(best_weighted_accuracies)
    mean_mpt = sum(mpts) / len(mpts)
    mean_w_mpt = sum(w_mpts) / len(w_mpts)
    
    for case in cases:
        if subset_roberta_match_predex:
            if case.roberta_prediction != case.case_label:
                continue
        sample_weights.extend(case.get_span_lengths())
        
        y_preds.extend(case.get_rot_labels())
        y_actual.extend(case.get_human_labels())
    
    # Compute overall AUC and accuracy
    try:
        overall_auc = sc_score(y_actual, y_preds)
        overall_weighted_auc = sc_score(y_actual, y_preds, sample_weight=sample_weights)
        auc_computable = True
    except ValueError:
        overall_auc = "NA"
        overall_weighted_auc = "NA"
        auc_computable = False
    
    binary_preds = [1 if p > 0.5 else 0 for p in y_preds]
    overall_accuracy = accuracy_score(y_actual, binary_preds)
    overall_weighted_accuracy = accuracy_score(y_actual, binary_preds, sample_weight=sample_weights)
    
    if verbose:
        # Original verbose output
        print(f'TOTAL NUMBER OF CASES: {case_count=}')
        print('======================')
        print(f'{subset_roberta_match_predex=}\tunweighted mean {get_metric_name()}:\t ', unweighted_mean_auc)
        print(f'{subset_roberta_match_predex=}\tunweighted mean Accuracy:', unweighted_mean_accuracy)
        print(f'{subset_roberta_match_predex=}\tweighted mean {get_metric_name()}:\t ', weighted_mean_auc)
        print(f'{subset_roberta_match_predex=}\tweighted mean Accuracy:\t ', weighted_mean_accuracy)
        print(f'{subset_roberta_match_predex=}\tBEST mean Accuracy\t\t:\t ', best_mean_accuracy)
        print(f'{subset_roberta_match_predex=}\tBEST weighted mean Accuracy\t:\t ', best_weighted_mean_accuracy)
        print('======================')
        
        if auc_computable:
            print(f"{subset_roberta_match_predex=}\t{get_metric_name()} \t\t:\t {overall_auc:.8f}")
            print(f"{subset_roberta_match_predex=}\tw_{get_metric_name()} \t\t:\t {overall_weighted_auc:.8f}")
        else:
            print(f"{subset_roberta_match_predex=}\t{get_metric_name()}: Cannot be computed (possibly due to single class in y_true).")
        print(f"{subset_roberta_match_predex=}\tAccuracy \t:\t {overall_accuracy:.8f}")
        print(f"{subset_roberta_match_predex=}\tw_Accuracy \t:\t {overall_weighted_accuracy:.8f}")
        print()
    else:
        # CSV output format
        csv_values = [
            index,
            comment,
            # formula,
            subset_roberta_match_predex,
            case_count,
            # unweighted_mean_auc,
            # unweighted_mean_accuracy,
            weighted_mean_auc,
            # weighted_mean_accuracy,
            # best_mean_accuracy,
            # best_weighted_mean_accuracy,
            # overall_auc,
            # overall_weighted_auc,
            # overall_accuracy,
            # overall_weighted_accuracy,
            # mean_mpt,
            # mean_w_mpt
        ]
        print(','.join(str(val) for val in csv_values))

import matplotlib.pyplot as plt
import numpy as np
import os

def span_stats(cases, flip_axes=False):
    human_annotations = []
    importances = []
    span_lengths = []
    proportions = []
    case_lengths = []
    quality = []
    sign_correctness = []
    
    for case in cases:
        human_annotations.extend(case.annotations)
        importances.extend(case.rot_preds)
        span_lengths.extend(case.get_span_lengths())
        case_length = sum(case.get_span_lengths())
        props = [l/case_length for l in case.get_span_lengths()]
        proportions.extend(props)
        case_lengths.extend([case_length] * len(case.get_span_lengths()))
        qualities = None
        if case.roberta_prediction > 0:
            qualities = [ann - pred for ann, pred in zip(case.annotations, case.rot_preds)]
        else:
            qualities = [ann - -pred for ann, pred in zip(case.annotations, case.rot_preds)]
        quality.extend(qualities)
        signs = None
        if case.roberta_prediction > 0:
            signs = [bool((ann == 1 and pred > 0) or (ann == 0 and pred < 0)) for ann, pred in zip(case.annotations, case.rot_preds)]
        else:
            signs = [bool((ann == 1 and pred < 0) or (ann == 0 and pred > 0)) for ann, pred in zip(case.annotations, case.rot_preds)]
        sign_correctness.extend(signs)

    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Y-axis variables to plot
    y_variables = {
        # 'human_annotations': human_annotations,
        'span_lengths': span_lengths,
        'proportions': proportions,
        'case_lengths': case_lengths
    }
    
    # X-axis variables
    x_variables = {
        'importances': importances,
        'sign_correctness': sign_correctness,
        'quality': quality
    }
    
    # Flip axes if requested
    if flip_axes:
        x_variables, y_variables = y_variables, x_variables
    
    # Create plots for each combination
    for x_name, x_data in x_variables.items():
        for y_name, y_data in y_variables.items():
            plt.figure(figsize=(10, 6))
            
            if x_name == 'sign_correctness':
                # For boolean x-axis, create box plots or violin plots
                correct_y = [y for x, y in zip(x_data, y_data) if x]
                incorrect_y = [y for x, y in zip(x_data, y_data) if not x]
                
                plt.boxplot([incorrect_y, correct_y], labels=['Incorrect', 'Correct'])
                plt.xlabel('Sign Correctness')
                plt.title(f'{y_name.replace("_", " ").title()} by Sign Correctness')
                
            elif y_name == 'sign_correctness' and flip_axes:
                # For boolean y-axis when axes are flipped
                correct_x = [x for y, x in zip(y_data, x_data) if y]
                incorrect_x = [x for y, x in zip(y_data, x_data) if not y]
                
                plt.boxplot([incorrect_x, correct_x], labels=['Incorrect', 'Correct'])
                plt.ylabel('Sign Correctness')
                plt.title(f'Sign Correctness by {x_name.replace("_", " ").title()}')
                
            else:
                # For continuous axes, create scatter plots
                plt.scatter(x_data, y_data, alpha=0.6)
                plt.xlabel(x_name.replace('_', ' ').title())
                plt.title(f'{y_name.replace("_", " ").title()} vs {x_name.replace("_", " ").title()}')
                
                # Add trend line
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                plt.plot(x_data, p(x_data), "r--", alpha=0.8)
            
            plt.ylabel(y_name.replace('_', ' ').title())
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            filename = f'plots/{y_name}_vs_{x_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {filename}")
    
    axis_mode = "flipped" if flip_axes else "default"
    print(f"All plots saved to 'plots/' directory with {axis_mode} axes")
    
    # Return the computed statistics for further analysis if needed
    return {
        'human_annotations': human_annotations,
        'importances': importances,
        'span_lengths': span_lengths,
        'proportions': proportions,
        'case_lengths': case_lengths,
        'quality': quality,
        'sign_correctness': sign_correctness
    }


def main():
    global FORMULA, METRIC
    parser = argparse.ArgumentParser(description="Prepare CASE objects from input files.")
    parser.add_argument("--directory", required=True, type=str, help="Subdirectory inside ./DATA/")
    parser.add_argument("--formula", required=True, type=str, default=' ', choices=["linear", "positived", "conditional", "flip", "flip_clip"], help="importance to prob formula")
    parser.add_argument("--comment", required=False, type=str, default=' ', help="Optional comment for CSV")
    parser.add_argument("--metric", required=False, type=str, default='auc', choices=["auc", "pr_auc"], help="Scoring metric: auc (ROC AUC) or pr_auc (average precision)")
    # parser.add_argument("--csv", default=False, type=bool, help="Set False for CSV style output")
    parser.add_argument("--csv", default=False, action='store_true', help="Output in CSV format")
    args = parser.parse_args()

    FORMULA = args.formula
    METRIC = args.metric

    base_dir = os.path.join("DATA", args.directory)
    # print(f"Looking inside: {base_dir}")

    # All subdirectories (assumes individual cases are within subfolders)
    all_csv_files = []
    meta_path = os.path.join(base_dir, "meta.csv")

    subdirs = [base_dir]
    if 'SHAP' not in args.directory and 'LIME' not in args.directory and 'GAUSSIAN' not in args.directory and 'UNIFORM' not in args.directory and 'IG' not in args.directory:
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    else:
        meta_path = os.path.join('DATA/SPAN_EXPS_stride_35_128_1e-05', 'meta.csv')

    for subdir in sorted(subdirs):
        csvs = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".csv") and f != "meta.csv"]
        all_csv_files.extend(sorted(csvs))

    # Read metadata for each case (RoT only)
    meta_df = pd.read_csv(meta_path)

    cases = []

    for filepath in all_csv_files:
        filename = os.path.basename(filepath)[:-4]  # drop .csv
        if filename not in meta_df['index'].values:
            # print(f"Warning: {filename} not in meta.csv — skipping.")
            continue

        # Metadata for this case
        row = meta_df[meta_df['index'] == filename].iloc[0]

        # Load span-level CSV
        csv_df = pd.read_csv(filepath)
        span_indices = csv_df['Unnamed: 0'].astype(str).tolist()
        #print(filepath)
        #print(csv_df)
        #print(span_indices)
        #break

        case = Case()
        case.case_name = filename
        case.case_label = row['ground_truth_judgement']
        case.roberta_prediction = row['roberta_prediction']
        case.rot_prediction = row['rot_prediction']
        case.is_not_rot = 'SHAP' in args.directory or 'GAUSSIAN' in args.directory and 'UNIFORM' in args.directory

        for i, span in enumerate(span_indices):
            is_annotated = int(span.startswith('___'))
            importance = csv_df['scaled_importance'].iloc[i]
            length = len(span.replace('___', '', 1))  # remove prefix when counting

            case.annotations.append(int(is_annotated))
            case.rot_preds.append(float(importance)/100.0)
            case.lengths.append(int(length))

        cases.append(case)

    # print(f"Loaded {len(cases)} cases.")
    # show(cases)
    # span_stats(cases)
    # span_stats(cases, flip_axes=True)
    # return
    
    # print('------------------------------------------------------------------------------------------------------------')
    # experiment(cases, use_abs=False, use_relu=False, use_roberta=True, flip_0=False, flip_1=False, subset_roberta_match_predex=False, index=args.directory, verbose=not bool(args.csv), comment=args.comment)
    # experiment(cases, use_abs=False, use_relu=False, use_roberta=True, flip_0=False, flip_1=False, subset_roberta_match_predex=True, index=args.directory, verbose=not bool(args.csv), comment=args.comment)
    


    # experiment(cases, subset_roberta_match_predex=False, index=f'{args.formula}___{args.directory}', verbose=not bool(args.csv), comment=args.comment)
    
    experiment(cases, subset_roberta_match_predex=True, index=f'{args.directory}', verbose=not bool(args.csv), comment=args.comment, formula=args.formula)
    # print('------------------------------------------------------------------------------------------------------------')
    # print()
    # print()
    # print()
    return


if __name__ == '__main__':
    main()

