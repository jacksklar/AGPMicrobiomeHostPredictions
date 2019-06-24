#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:47:43 2019

@author: sklarjg
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from scipy import interp
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef


metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
otu_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Otu_Data.csv", index_col = 0)
feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)
frequency_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/frequency_feature_info.csv", index_col = 0)
taxa_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/taxa_md5.xls", sep = "\t", index_col = 0)
taxa_df = taxa_df[taxa_df.index.isin(otu_df.columns)]
taxa_df = taxa_df.replace(np.nan, 'Unknown', regex=True)
otu_df = otu_df.loc[metadata_df.index, :]

### LOAD MICROBIOME OTU EXPRESSION DATA
data = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/HIV/Data/seqtab_md5_150nt_6min_rar10k.txt", sep = "\t")
data.index = data["OTU ID"]
data.drop("OTU ID", axis = 1, inplace = True)
data = data.T
data.sort_index(inplace=True)

### LOAD PATIENT METADATA
metadata = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/HIV/Data/180904_Dutch samples manifest_IVC3_mapforR.txt", sep = "\t")
metadata.index = metadata["sampID"]
metadata.sort_index(inplace=True)
###REMOVE METADATA WITH NO SAMPLES
remove_from_meta = np.setdiff1d(metadata.index, data.index)
metadata.drop(remove_from_meta, axis = 0, inplace = True)
###REMOVE SAMPLES WITH NO METADATA
remove_from_data = np.setdiff1d(data.index, metadata.index)
data.drop(remove_from_data, axis = 0, inplace = True)
data_ind = metadata.loc[metadata["sampID"].isin(data.index),"UserNumber"]
metadata.index = metadata["UserNumber"]
data.index = data_ind
data.sort_index(inplace=True)
metadata.sort_index(inplace=True)
msm = metadata[metadata["all_MSMnonMSM"] == "MSM"].index.values
nonmsm = metadata[metadata["all_MSMnonMSM"] == "nonMSM"].index.values
fem = metadata[metadata["cat1"] == "FND"].index.values
metadata.loc[msm,"MSM"] = "MSM"
metadata.loc[nonmsm,"nonMSM"] = "nonMSM"
metadata.loc[fem,"FEM"] = "FEM"

agp_otus = otu_df.columns.values
hiv_otus = data.columns.values
union_otus = np.intersect1d(agp_otus, hiv_otus)
otu_hiv = data.loc[:, union_otus]
otu_agp = otu_df.loc[:, union_otus]

#%%

#otu_df_full = otu_df
#otu_max_abund =  otu_df.mean(axis = 0).div(10000)
#otu_sig = otu_max_abund[otu_max_abund > 0.0001].index
#otu_df = otu_df.loc[:, otu_sig]
#print otu_df.shape[1]

#otu_max_abund = data.div(data.sum(axis = 1), axis = 0).max(axis = 0)
#otu_filter_out = otu_max_abund[otu_max_abund < 0.001].index#
#data = data.drop(otu_filter_out, axis = 1)

#%%

dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/binary_cohorts/"
hiv_metadata = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/HIV/Data/microbiome_incl_sexbehavior.csv", index_col = 0)


def buildDataSubset(otu_data, cohort):
        X = otu_data.loc[cohort.index,:].astype(float).values
        y = cohort["target"].astype(float)
        print y.value_counts()
        X = np.log(X + 1.0)
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        X, y = shuffle(X, y)
        if len(X) > 2000:
            X = X[:2000,:]
            y = y[:2000]
        return X, y        

def HyperparameterOpt(X, y):
        print "Hyperparameter Optimization: 100 iterations of 4-fold cv random search"
        parameters_rf = {'max_depth': [10, 20, 40, 60, 80, 100, None],
                         'min_samples_leaf': [1, 2],
                         'min_samples_split': [2, 4],
                         'n_estimators': [100, 150, 200, 300, 400]}
        rf  = RandomForestClassifier(class_weight = 'balanced')
        rf_random = RandomizedSearchCV(estimator = rf, 
                                       param_distributions = parameters_rf, 
                                       n_iter = 100, 
                                       cv = 4, 
                                       verbose=2, 
                                       random_state=1231242, 
                                       n_jobs = -1)
        rf_random.fit(X, y)
        opt_params = rf_random.best_params_
        print opt_params
        return opt_params

alc_day_cohort = pd.read_csv(dir_path + "alcohol_daily_cohort.csv", index_col = 0)
#alc_reg_cohort = dir_path + "alcohol_regular_cohort.csv"
#alc_occ_cohort = dir_path + "alcohol_occasional_cohort.csv"
#alc_rare_cohort = dir_path + "alcohol_rare_cohort.csv"


X, y = buildDataSubset(otu_agp, alc_day_cohort)
opt_params = HyperparameterOpt(X, y)
alg = RandomForestClassifier(class_weight = 'balanced').set_params(**opt_params)
alg_shuf = RandomForestClassifier(class_weight = 'balanced').set_params(**opt_params)
        
alg.fit(X, y)   
imp = alg.feature_importances_             

#%%
hiv_metadata = hiv_metadata.drop_duplicates(subset= "UserNumber")
otu_hiv = otu_hiv.drop_duplicates()
hiv_metadata.index = hiv_metadata["UserNumber"]
alc_test_inds = hiv_metadata[hiv_metadata["alcofreq"].isin(["Daily or nearly daily ", "not drinking"])].index
otu_test = otu_hiv.loc[alc_test_inds,:]
alc_test= hiv_metadata.loc[alc_test_inds, "alcofreq"].map({'Daily or nearly daily ': 1,'not drinking': 0})

#%%


probas_ = alg.predict_proba(otu_test)
y_pred = probas_[:,1]
y_true = alc_test
        
compare = np.vstack([y_true, y_pred]).T

#%%


#probas_ = alg.predict_proba(X)
#y_pred = probas_[:,1]
#y_true = y        
#compare = np.vstack([y_true, y_pred]).T


#%%

"""
        #cv = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.3)        
        #for fold_num, (train, test) in enumerate(cv.split(X, y)):    
            #print fold_num,
            alg.fit(X[train], y[train])   
            probas_ = alg.predict_proba(X[test])
            y_pred = probas_[:,1]
            y_true = y[test]
            imp = alg.feature_importances_        
            #imp = alg.coef_[0,:]
            
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            self.importances.append(imp)
            self.tprs.append(interp(self.mean_fpr, fpr, tpr))
            self.tprs[-1][0] = 0.0
            self.aucs.append(roc_auc)
            
            y_shuffled = shuffle(y)   
            alg_shuf.fit(X[train], y_shuffled[train])   
            probas_ = alg_shuf.predict_proba(X[test])
            y_pred = probas_[:,1]
            y_true = y_shuffled[test]
            
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            self.shuffled_tprs.append(interp(self.mean_fpr, fpr, tpr))
            self.shuffled_tprs[-1][0] = 0.0
            self.shuffled_aucs.append(roc_auc)
            

"""


















