# Utils
import time
import torch
import numpy as np
import pickle
import os
import argparse  # Step 1: Import argparse
import warnings
import pandas as pd

# ML models
from openxai.LoadModel import LoadModel

# Data loaders
from openxai.dataloader import return_loaders

# Explanation models
from openxai.Explainer import Explainer

# Evaluation methods
from openxai.evaluator import Evaluator

# Perturbation methods required for the computation of the relative stability metrics
from openxai.explainers.catalog.perturbation_methods import NormalPerturbation
from openxai.explainers.catalog.perturbation_methods import NewDiscrete_NormalPerturbation
from tqdm import tqdm

# Step 2: Create argument parsers
parser = argparse.ArgumentParser(description="Generate explanations for ML models")
parser.add_argument("--model_name", type=str, choices=["lr", "ann"], help="Name of the ML model")
parser.add_argument("--data_name", type=str, choices=["heloc", "adult", "german", "compas"], help="Name of the dataset")
args = parser.parse_args()
WITH_BASELINE = True

# Use the values from command-line arguments
model_name = args.model_name
data_name = args.data_name

# Load pretrained ml model
model = LoadModel(data_name=data_name, ml_model=model_name, pretrained=True)

# Get training and test loaders
loader_train, loader_test = return_loaders(data_name=data_name, download=True, batch_size=1, shuffle=False)


data_all = torch.FloatTensor(loader_train.dataset.data)

# Initialize all explainers
methods = ['grad', 'sg', 'itg', 'ig', 'shap', 'rot', 'lime', 'rotrbf']
methods = ['ig', 'shap', 'rot', 'lime']#, 'sg']
# methods = ['rot']#, 'sg']

# methods = ['lime']
# methods = ['rot']
# methods = ['shap']
# methods = ['ig']
# methods = ['itg']
# methods = ['sg']
# methods = ['grad']
explainers = {}
init_times = {}

all_exps = {method: [] for method in methods}
all_baselines = {method: [] for method in methods}

for method in methods:
    init_start = time.time()
    if model_name == 'ann':
        if method == 'lime':
            explainers['lime'] = Explainer(method=method, model=lambda explicand: model(torch.from_numpy(np.array(explicand)).float()).detach().numpy(), dataset_tensor=data_all)
        else:
            explainers[method] = Explainer(method=method, model=model, dataset_tensor=data_all)
        pass
    else:
        if method == 'lime':
            explainers['lime'] = Explainer(method=method, model=lambda explicand: model.linear(torch.from_numpy(np.array(explicand)).float()).detach().numpy(), dataset_tensor=data_all)
        else:
            explainers[method] = Explainer(method=method, model=model.linear, dataset_tensor=data_all)
    init_end = time.time()
    init_times[method] = init_end - init_start

output_dir = f"./DATA/EXPLAINER_OUTPUTS/EXPLANATIONS_{data_name}_{model_name}"
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
os.makedirs("./DATA/EXPLAINER_OUTPUTS/META", exist_ok=True)

total_times = {method: 0 for method in methods}

# Wrap the loop using tqdm
print(f"[START] Generating Explanations {data_name} {model_name}")
all_inputs = []
for i, data in enumerate(tqdm(loader_train, desc=f"Generating Explanations {data_name} {model_name}", ncols=100, dynamic_ncols=True)):
    # if i > 50:
    #     break
    inputs, labels = data
    all_inputs.append(inputs.detach().numpy().flatten())
    labels = labels.type(torch.int64)

    for method in methods:
        # Get explanation for the current data point
        time_temp = time.time()
        if not WITH_BASELINE:
            assert False
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)

                baseline, exp = torch.tensor(0), None
                if method in ['lime', 'rot', 'shap', 'ig']:
                    baseline, exp = explainers[method].get_explanation(inputs.float(), label=None, with_baseline = True)
                else:
                    assert False
            
            total_times[method] += time.time() - time_temp
            all_exps[method].append(exp.detach().numpy().flatten())
            all_baselines[method].append(np.array(baseline.detach().numpy()).flatten()[0])

        # Save the explanation to a file (e.g., as a PyTorch tensor)
        output_file = os.path.join(output_dir, f"{method}_{i}.pt")
        torch.save(exp, output_file)

all_inputs = pd.DataFrame(all_inputs, columns=loader_train.dataset.feature_names)
all_inputs.to_csv(f'./DATA/EXPLAINER_OUTPUTS/META/{data_name}_input_tensors.csv')

for method in methods:
    df = pd.DataFrame(np.array(all_exps[method]), columns=loader_train.dataset.feature_names)
    df.to_csv(f'./DATA/EXPLAINER_OUTPUTS/META/exp_{data_name}_{model_name}_{method}.csv')
    df = pd.DataFrame(np.array(all_baselines[method]).flatten(), columns=['baseline'])
    df.to_csv(f'./DATA/EXPLAINER_OUTPUTS/META/base_{data_name}_{model_name}_{method}.csv')

print(f"[END] Generating Explanations {data_name} {model_name}")
print()

with open(f"./DATA/EXPLAINER_OUTPUTS/META/TIMES_{data_name}_{model_name}.txt", "w") as file:
    for method in methods:
        file.write(f"Init {method.upper()} Time: {init_times[method]}\n")
        file.write(f"{method.upper()} Time: {total_times[method]}\n")
        file.write(f"Total {method.upper()} Time: {total_times[method] + init_times[method]}\n")
