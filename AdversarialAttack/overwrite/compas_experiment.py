"""
The experiment MAIN for COMPAS.
 * Run the file and the COMPAS experiments will complete
 * This may take some time because we iterate through every instance in the test set for
   both LIME and SHAP explanations take some time to compute
 * The print outs can be interpreted as maps from the RANK to the rate at which the feature occurs in the rank.. e.g:
 	    1: [('length_of_stay', 0.002592352559948153), ('unrelated_column_one', 0.9974076474400518)]
   can be read as the first unrelated column occurs ~100% of the time in as the most important feature
 * "Nothing shown" refers to SHAP yielding only 0 shapley values 
"""
import warnings

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from copy import deepcopy
import json
import time

from rule_of_thumb import RuleOfThumb

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_compas_data(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])
features = [c for c in X]

race_indc = features.index('race')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

X = X.values

print (features)
###
## The models f and psi for COMPAS.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X]))

# the display model with two unrelated features
class innocuous_model_psi_two:
	def predict_proba(self, X):
		A = np.where(X[:,unrelated_indcs] > 0, params.positive_outcome, params.negative_outcome)
		B = np.where(X[:,unrelated_indcs1] > 0, params.positive_outcome, params.negative_outcome)
		preds = np.logical_xor(A, B).astype(int)
		return one_hot_encode(preds)
#
##
###

def experiment_main():
	"""
	Run through experiments for LIME/SHAP on compas using both one and two unrelated features.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1, random_state=0)
	ss = StandardScaler().fit(xtrain)
	xtrain = ss.transform(xtrain)
	xtest = ss.transform(xtest)

	print ('---------------------')
	print ("Beginning LIME COMPAS Experiments....")
	print ("(These take some time to run because we have to generate explanations for every point in the test set) ") # 'two_year_recid','c_charge_degree'
	print ('---------------------')

	# Train the adversarial model for LIME with f and psi 
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'), features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], feature_names=features, perturbation_multiplier=30)
	lime1_t1 = time.time()
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, sample_around_instance=True, feature_names=adv_lime.get_column_names(), categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'),features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], discretize_continuous=False)
	lime1_t2 = time.time()
	explanations = []
	store_results1 = []
	df_rows = []
	for i in range(xtest.shape[0]):
		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())
		df_rows.append({_ft.split('=')[0]: _imp for _ft, _imp in explanations[-1]})
		store_results1.append(experiment_summary([explanations[-1]], features))
	lime1_t3 = time.time()
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/compas/lime_exps_ml1.csv')

	stored_res1 = json.dumps(store_results1)
	with open('./RESULTS/compas/Compas_LIME_one.json', 'w') as file:
		file.write(stored_res1)

	# Display Results
	print ("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
	print (experiment_summary(explanations, features))
	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

	xx = xtrain
	lrot1_t1 = time.time()
	yy = adv_lime.predict(xx)
	rot1 = RuleOfThumb(yy, xx)
	lrot1_t2 = time.time()
	xx = xtest
	rot_exps1 = rot1.get_explanation(xx)
	lrot1_t3 = time.time()
	rot_results1 = []
	df_rows = []
	for exp in rot_exps1:
		_temp = [(_feat, _exp) for _feat, _exp in zip(features, exp)]
		df_rows.append({_ft: _imp for _ft, _imp in zip(features, exp)})
		rot_results1.append(experiment_summary([_temp], features))
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/compas/rot_exps_ml1.csv')
	stored_rot1 = json.dumps(rot_results1)
	with open('./RESULTS/compas/Compas_LIME_rot_one.json', 'w') as file:
		file.write(stored_rot1)
	
	# Repeat the same thing for two features
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'),features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], feature_names=features, perturbation_multiplier=30)
	lime2_t1 = time.time()
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'),features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], discretize_continuous=False)
	lime2_t2 = time.time()
	explanations = []
	store_result2 = []
	df_rows = []
	for i in range(xtest.shape[0]):
		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())
		df_rows.append({_ft.split('=')[0]: _imp for _ft, _imp in explanations[-1]})
		store_result2.append(experiment_summary([explanations[-1]], features))
	lime2_t3 = time.time()
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/compas/lime_exps_ml2.csv')

	stored_res2 = json.dumps(store_result2)
	with open('./RESULTS/compas/Compas_LIME_two.json', 'w') as file:
		file.write(stored_res2)

	print ("LIME Ranks and Pct Occurances two unrelated features:")
	print (experiment_summary(explanations, features))
	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

	xx = xtrain
	lrot2_t1 = time.time()
	yy = adv_lime.predict(xx)
	rot2 = RuleOfThumb(yy, xx)
	lrot2_t2 = time.time()
	xx = xtest
	rot_exps2 = rot2.get_explanation(xx)
	lrot2_t3 = time.time()
	rot_results2 = []
	df_rows = []
	for exp in rot_exps2:
		_temp = [(_feat, _exp) for _feat, _exp in zip(features, exp)]
		df_rows.append({_ft: _imp for _ft, _imp in zip(features, exp)})
		rot_results2.append(experiment_summary([_temp], features))
	stored_rot2 = json.dumps(rot_results2)
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/compas/rot_exps_ml2.csv')
	with open('./RESULTS/compas/Compas_LIME_rot_two.json', 'w') as file:
		file.write(stored_rot2)
	
	
	print ('---------------------')
	print ('Beginning SHAP COMPAS Experiments....')
	print ('---------------------')

	#Setup SHAP
	background_distribution = shap.kmeans(xtrain,10)
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features)
	shap1_t1 = time.time()
	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	shap1_t2 = time.time()
	explanations = adv_kerenel_explainer.shap_values(xtest)
	shap1_t3 = time.time()

	# format for display
	formatted_explanations = []
	store_result3 = []
	df_rows = []
	df_rows = []
	for exp in explanations:
		formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
		df_rows.append({_ft: _imp for _ft, _imp in zip(features, exp)})
		store_result3.append(experiment_summary([formatted_explanations[-1]], features))
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/compas/shap_exps_ms1.csv')
	
	stored_res3 = json.dumps(store_result3)
	with open('./RESULTS/compas/Compas_SHAP_one.json', 'w') as file:
		file.write(stored_res3)
	
	print ("SHAP Ranks and Pct Occurances one unrelated features:")
	print (experiment_summary(formatted_explanations, features))
	print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

	xx = xtrain
	rot1_t1 = time.time()
	yy = adv_shap.predict(xx)
	rot3 = RuleOfThumb(yy, xx)
	rot1_t2 = time.time()
	xx = xtest
	rot_exps3 = rot3.get_explanation(xx)
	rot1_t3 = time.time()
	rot_results3 = []
	df_rows = []
	for exp in rot_exps3:
		_temp = [(_feat, _exp) for _feat, _exp in zip(features, exp)]
		df_rows.append({_ft: _imp for _ft, _imp in zip(features, exp)})
		rot_results3.append(experiment_summary([_temp], features))
	stored_rot3 = json.dumps(rot_results3)
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/compas/rot_exps_ms1.csv')
	with open('./RESULTS/compas/Compas_SHAP_rot_one.json', 'w') as file:
		file.write(stored_rot3)	

	background_distribution = shap.kmeans(xtrain,10)
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, feature_names=features)
	shap2_t1 = time.time()
	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	shap2_t2 = time.time()
	explanations = adv_kerenel_explainer.shap_values(xtest)
	shap2_t3 = time.time()

	# format for display
	formatted_explanations = []
	store_results4 = []
	df_rows = []
	for exp in explanations:
		formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
		df_rows.append({_ft: _imp for _ft, _imp in zip(features, exp)})
		store_results4.append(experiment_summary([formatted_explanations[-1]], features))

	stored_res4 = json.dumps(store_results4)
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/compas/shap_exps_ms2.csv')
	with open('./RESULTS/compas/Compas_SHAP_two.json', 'w') as file:
		file.write(stored_res4)

	print ("SHAP Ranks and Pct Occurances two unrelated features:")
	print (experiment_summary(formatted_explanations, features))
	print ("Fidelity:",round(adv_shap.fidelity(xtest),2))
	print ('---------------------')
	
	xx = xtrain
	rot2_t1 = time.time()
	yy = adv_shap.predict(xx)
	rot4 = RuleOfThumb(yy, xx)
	rot2_t2 = time.time()
	xx = xtest
	rot_exps4 = rot4.get_explanation(xx)
	rot2_t3 = time.time()
	rot_results4 = []
	df_rows = []
	for exp in rot_exps4:
		_temp = [(_feat, _exp) for _feat, _exp in zip(features, exp)]
		df_rows.append({_ft: _imp for _ft, _imp in zip(features, exp)})
		rot_results4.append(experiment_summary([_temp], features))
	stored_rot4 = json.dumps(rot_results4)
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/compas/rot_exps_ms2.csv')
	with open('./RESULTS/compas/Compas_SHAP_rot_two.json', 'w') as file:
		file.write(stored_rot4)

	with open('./RESULTS/compas/compas_timestamps.txt', 'w') as file:
		file.write(f'shap1 time [init]: {shap1_t2 - shap1_t1}\n')
		file.write(f'shap1 time [total]: {shap1_t3 - shap1_t1}\n')
		file.write(f'rot1 time [init]: {rot1_t2 - rot1_t1}\n')
		file.write(f'rot1 time [total]: {rot1_t3 - rot1_t1}\n')
		file.write(f'shap2 time [init]: {shap2_t2 - shap2_t1}\n')
		file.write(f'shap2 time [total]: {shap2_t3 - shap2_t1}\n')
		file.write(f'rot2 time [init]: {rot2_t2 - rot2_t1}\n')
		file.write(f'rot2 time [total]: {rot2_t3 - rot2_t1}\n')
		file.write(f'\ntotal number of explanations: {len(explanations)}\n\n\n')
		file.write(f'lime1 time [init]: {lime1_t2 - lime1_t1}\n')
		file.write(f'lime1 time [total]: {lime1_t3 - lime1_t1}\n')
		file.write(f'lrot1 time [init]: {lrot1_t2 - lrot1_t1}\n')
		file.write(f'lrot1 time [total]: {lrot1_t3 - lrot1_t1}\n')
		file.write(f'lime2 time [init]: {lime2_t2 - lime2_t1}\n')
		file.write(f'lime2 time [total]: {lime2_t3 - lime2_t1}\n')
		file.write(f'lrot2 time [init]: {lrot2_t2 - lrot2_t1}\n')
		file.write(f'lrot2 time [total]: {lrot2_t3 - lrot2_t1}\n')
		file.write(f'\ntotal number of explanations: {len(explanations)}\n')

if __name__ == "__main__":
	experiment_main()
