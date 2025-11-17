"""
The experiment MAIN for GERMAN.
"""
import warnings
warnings.filterwarnings('ignore') 

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

from sklearn.cluster import KMeans 

from copy import deepcopy
import json
import time

from rule_of_thumb import RuleOfThumb

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_german(params)

features = [c for c in X]

gender_indc = features.index('Gender')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')

X = X.values

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1, random_state=0)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

mean_lrpi = np.mean(xtrain[:,loan_rate_indc])


categorical = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone','CheckingAccountBalance_geq_0','CheckingAccountBalance_geq_200','SavingsAccountBalance_geq_100','SavingsAccountBalance_geq_500','MissedPayments','NoCurrentLoan','CriticalAccountOrLoansElsewhere','OtherLoansAtBank','OtherLoansAtStore','HasCoapplicant','HasGuarantor','OwnsHouse','RentsHouse','Unemployed','YearsAtCurrentJob_lt_1','YearsAtCurrentJob_geq_4','JobClassIsSkilled']
categorical = [features.index(c) for c in categorical]

###
## The models f and psi for GERMAN.  We discriminate based on gender for f and consider loan rate % income for explanation
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negative outcome if female
    def predict(self,X):
        return np.array([params.negative_outcome if x[gender_indc] < 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to loan rate indc
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[loan_rate_indc] > mean_lrpi else params.positive_outcome for x in X]))

##
###

def experiment_main():
	"""
	Run through experiments for LIME/SHAP on GERMAN.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	print ('---------------------')
	print ("Beginning LIME GERMAN Experiments....")
	print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
	print ('---------------------')

	# Train the adversarial model for LIME with f and psi 
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=30, categorical_features=categorical)
	lime_t1 = time.time()
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False, categorical_features=categorical)
	lime_t2 = time.time()
	explanations = []
	store_results1 = []
	df_rows = []
	for i in range(xtest.shape[0]):
		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())
		df_rows.append({_ft.split('=')[0]: _imp for _ft, _imp in explanations[-1]})
		store_results1.append(experiment_summary([explanations[-1]], features))
	lime_t3 = time.time()
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/german/lime_exps_ml.csv')

	stored_res1 = json.dumps(store_results1)
	with open('./RESULTS/german/German_LIME.json', 'w') as file:
		file.write(stored_res1)

	xx = xtrain
	lrot_t1 = time.time()
	yy = adv_lime.predict(xx)
	rot1 = RuleOfThumb(yy, xx)
	lrot_t2 = time.time()
	xx = xtest
	rot_exps1 = rot1.get_explanation(xx)
	lrot_t3 = time.time()
	rot_results1 = []
	df_rows = []
	for exp in rot_exps1:
		_temp = [(_feat, _exp) for _feat, _exp in zip(features, exp)]
		df_rows.append({_ft: _imp for _ft, _imp in zip(features, exp)})
		rot_results1.append(experiment_summary([_temp], features))
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/german/rot_exps_ml.csv')
	stored_rot1 = json.dumps(rot_results1)
	with open('./RESULTS/german/German_LIME_rot.json', 'w') as file:
		file.write(stored_rot1)
	
	# Display Results
	print ("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
	print (experiment_summary(explanations, features))
	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

	
	print ('---------------------')
	print ('Beginning SHAP GERMAN Experiments....')
	print ('---------------------')

	#Setup SHAP
	background_distribution = KMeans(n_clusters=10,random_state=0).fit(xtrain).cluster_centers_
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, 
			feature_names=features, background_distribution=background_distribution, rf_estimators=100, n_samples=5e4)
	shap_t1 = time.time()
	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution,)
	shap_t2 = time.time()
	explanations = adv_kerenel_explainer.shap_values(xtest)
	shap_t3 = time.time()
	
	# format for display
	formatted_explanations = []
	store_result3 = []
	df_rows = []
	for exp in explanations:
		formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
		df_rows.append({_ft: _imp for _ft, _imp in zip(features, exp)})
		store_result3.append(experiment_summary([formatted_explanations[-1]], features))
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/german/shap_exps_ms.csv')
	
	stored_res3 = json.dumps(store_result3)
	with open('./RESULTS/german/German_SHAP_one.json', 'w') as file:
		file.write(stored_res3)

	print ("SHAP Ranks and Pct Occurances one unrelated features:")
	print (experiment_summary(formatted_explanations, features))
	print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

	xx = xtrain
	rot_t1 = time.time()
	yy = adv_shap.predict(xx)
	rot3 = RuleOfThumb(yy, xx)
	rot_t2 = time.time()
	xx = xtest
	rot_exps3 = rot3.get_explanation(xx)
	rot_t3 = time.time()
	rot_results3 = []
	df_rows = []
	for exp in rot_exps3:
		_temp = [(_feat, _exp) for _feat, _exp in zip(features, exp)]
		df_rows.append({_ft: _imp for _ft, _imp in zip(features, exp)})
		rot_results3.append(experiment_summary([_temp], features))
	stored_rot3 = json.dumps(rot_results3)
	df = pd.DataFrame(df_rows)
	df.to_csv('./RESULTS/german/rot_exps_ms.csv')

	with open('./RESULTS/german/German_SHAP_rot.json', 'w') as file:
		file.write(stored_rot3)	
	print ('---------------------')

	with open('./RESULTS/german/german_timestamps.txt', 'w') as file:
		file.write(f'shap time [init]: {shap_t2 - shap_t1}\n')
		file.write(f'shap time [total]: {shap_t3 - shap_t1}\n')
		file.write(f'rot time [init]: {rot_t2 - rot_t1}\n')
		file.write(f'rot time [total]: {rot_t3 - rot_t1}\n')
		file.write(f'\ntotal number of explanations: {len(explanations)}\n\n\n')
		file.write(f'lime time [init]: {lime_t2 - lime_t1}\n')
		file.write(f'lime time [total]: {lime_t3 - lime_t1}\n')
		file.write(f'lrot time [init]: {lrot_t2 - lrot_t1}\n')
		file.write(f'lrot time [total]: {lrot_t3 - lrot_t1}\n')
		file.write(f'\ntotal number of explanations: {len(explanations)}\n')



if __name__ == "__main__":
	experiment_main()
