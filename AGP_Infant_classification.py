#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:24:21 2019

@author: sklarjg
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import xgboost as xgb

import scipy.stats as stats
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp

frequency_map = {0: "Never",    
                 1: "Rarely (a few times/month)",
                 2: "Occasionally (1-2 times/week)",
                 3: "Regularly (3-5 times/week)",
                 4: "Daily",
                 5: "N/A"}

binary_map = {0: "No",    
              1: "Yes",
              2: "N/A"}

disease_map = {0: "I do not have this condition",
               1: "Diagnosed by a medical professional (doctor, physician assistant)",
               2: "Diagnosed by an alternative medicine practitioner",    
               3: "Self-diagnosed",
               4: "N/A"}



#feature_groups = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_groups.csv")
#metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv")
#otu_unique_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Otu_Data.csv")


#metadata_vioscreen_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/VioScreen_Metadata.csv")
#metadata_temporal_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Temporal_Metadata.csv")
#vioscreen_data_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/VioScreen_Data.csv")
#otu_temporal_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Temporal_Otu_Data.csv")
#otu_vioscreen_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_VioScreen_Otu_Data.csv")


#%%

def discrete(a):
    if a > 2.0:
        a = "Adult"
    else:
        a = "Infant"
    return a


def KFoldRandForests_classification(X,y, n_split = 3, title = None, save = False, filepath = None):

    tprs = []                           ####
    aucs = []                              #
                                           ###=- empty arrays to save model performance metrics over folds of training 
    importances = []                       #
    plt.figure(1, figsize=(8, 8))
    mean_fpr = np.linspace(0, 1, 100)   ### continuous range for plotting average roc curve
    X,y = shuffle(X,y)  
    cv = StratifiedKFold(n_splits = n_split, random_state = 234236)

    ### TRAIN OVER N_SPLIT CROSS FOLDS:
    for fold_num, (train, test) in enumerate(cv.split(X, y)): 
        print "    Fold #%d" %(fold_num)
        ## DECISION TREE CLASSIFIER
        #clf = RandomForestClassifier(n_estimators = 256)
        #probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        #y_pred = probas_[:,1]
        #y_true = y[test]
        
        clf = xgb.XGBClassifier(learning_rate= 0.5, n_estimators= 512, silent = True, objective= 'binary:logitraw')
        clf.fit(X[train], y[train], verbose=True, eval_set=[(X[test], y[test])], early_stopping_rounds=50)
        y_pred = clf.predict_proba(X[test])
        y_pred = y_pred[:, 1]
        y_true = y[test]
 
        ###FEATURE IMPORTANCE VALUES FOR OTUS 
        imp = clf.feature_importances_
        importances.append(imp)
    
        ### COMPUTE ROC CURVE FOR CURRENT FOLD:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
    ### COMPUTE MEAN OTU IMPORTANCE OVER ALL FOLDS
    importances = np.mean(importances, axis=0)
    #importances_std = np.mean(importances_std, axis=0)
    
    ### PLOT AVERAGE ROC CUVRE WITH UNCERTAINTY AREA
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, 
             color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=0.9)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    plt.title(title)
    ### SAVE PLOT
    if save == True:
        plt.savefig(filepath, dpi = 300)
    plt.show()
    #return importances, importances_std
    


#%%
    
    

metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AGP_supp/Old_version/Data/Cohort_Subsets/AGP_infant_meta.csv", sep = ",", index_col = 0)
otu_df2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AGP_supp/Old_version/Data/Cohort_Subsets/AGP_infant_ots.csv", sep = ",", index_col = 0)



metadata_df["duptest"] = metadata_df['age_years'].astype(str)+'_'+metadata_df['bmi'].astype(str)+'_'+metadata_df['geo_loc_name']+'_'+metadata_df['probiotic_frequency']+'_'+metadata_df['height_cm'].astype(str)
metadata_df2 = metadata_df.drop_duplicates(subset='duptest', keep="first")


num_infant = len(metadata_df2[metadata_df2["age_years"] < 3.0 ])
num_adult = len(metadata_df2[metadata_df2["age_years"] > 3.0])

adult_remove = metadata_df2[metadata_df2["age_years"] > 3.0].sample(num_adult - num_infant).index
metadata_df2 = metadata_df2[~metadata_df2.index.isin(adult_remove)]

otu_df2 = otu_df2.loc[metadata_df2.index, :]


duplicates =metadata_df[metadata_df.duplicated(subset="duptest", keep=False)]

#%%

y = metadata_df2.loc[otu_df2.index, "age_years"]
y = y.apply(discrete)
X = otu_df2.astype(float)
X, y = shuffle(X, y)
X = X.values
le = LabelEncoder()
le.fit(y)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
y = le.transform(y) 

data_dmatrix = xgb.DMatrix(data=X,label=y)

KFoldRandForests_classification(X, 
                                y, 
                                n_split = 5, 
                                title = "Microbiome of Infant vs. Adult", 
                                save = False, 
                                filepath = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Infant/InfvsAdult.png")





#%%


print y.value_counts()













