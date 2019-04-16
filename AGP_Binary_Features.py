#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:17:44 2019

@author: sklarjg
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import seaborn as sns        


from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold,  KFold, RepeatedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, f1_score
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


'''
frequency_map = {0: "Never",    
                 1: "Rarely (a few times/month)",
                 2: "Occasionally (1-2 times/week)",
                 3: "Regularly (3-5 times/week)",
                 4: "Daily",
                 5: "N/A"}
'''

feature_groups = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_groups.csv", index_col = 0)
metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
otu_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Otu_Data.csv", index_col = 0)
feature_groups = feature_groups[~feature_groups.index.isin(['dna_extracted','physical_specimen_remaining','public', 'assigned_from_geo'])]
binary_features = feature_groups[feature_groups["group"].isin(["binary","disease"])].index.values
other_features = feature_groups[feature_groups["group"] == 'other'].index.values

otu_max_abund = otu_df.div(otu_df.sum(axis = 1), axis = 0).max(axis = 0).sort_values(ascending = False)
otu_filter_out = otu_max_abund[otu_max_abund < 0.01].index
otu_df = otu_df.drop(otu_filter_out, axis = 1)

metadata_df = metadata_df[metadata_df["subset_healthy"] == True]
otu_df = otu_df[otu_df.index.isin(metadata_df.index)]

#metadata_vioscreen_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/VioScreen_Metadata.csv")
#metadata_temporal_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Temporal_Metadata.csv")
#vioscreen_data_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/VioScreen_Data.csv")
#otu_temporal_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Temporal_Otu_Data.csv")
#otu_vioscreen_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_VioScreen_Otu_Data.csv")

taxa_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/taxa_md5.xls", sep = "\t", index_col = 0)
taxa_df = taxa_df[taxa_df.index.isin(otu_df.columns)]
taxa_df = taxa_df.replace(np.nan, 'Unknown', regex=True)
                          
#%%

###MATCHING METADATA FEATURES TO BUILD BALANCED COHORTS
le_race = LabelEncoder()
le_diet = LabelEncoder()
le_sex = LabelEncoder()

mid_america_log = -100
mid_america_lat = 40


metadata_df["age_years"] = metadata_df["age_years"].astype(float)
metadata_df["bmi"] = metadata_df["bmi"].astype(float)
metadata_df["weight_kg"] = metadata_df["weight_kg"].astype(float)
metadata_df["longitude"] = metadata_df["longitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)
metadata_df["latitude"] = metadata_df["latitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)

metadata_df["race"] = metadata_df["race"].replace(["Unspecified", "Not provided"], "Other")
le_race.fit(metadata_df["race"].unique())
metadata_df["race"] = le_race.transform(metadata_df["race"])

metadata_df["diet_type"] = metadata_df["diet_type"].replace(["Unspecified", "Not provided"], "Other")
le_diet.fit(metadata_df["diet_type"].unique())
metadata_df["diet_type"] = le_diet.transform(metadata_df["diet_type"])

metadata_df['sex'] = metadata_df['sex'].replace(["Unspecified", "Not provided", "unspecified"], "other")
le_sex.fit(metadata_df["sex"].unique())
metadata_df["sex"] = le_sex.transform(metadata_df["sex"])

cohort_matching_features = ["sex", "age_years", "bmi", "longitude", "latitude", "race"]
metadata_matching = metadata_df.loc[:, cohort_matching_features]

scaler = StandardScaler()
metadata_matching= pd.DataFrame(scaler.fit_transform(metadata_df.loc[:, cohort_matching_features].astype(float)), index = metadata_df.index, columns = cohort_matching_features)

#%%

class AGPCohortClassification:

    def __init__(self, target, optimize, save, title, filename):
        self.target = target
        self.optimize = optimize
        self.save = save
        self.title = title
        self.filename = filename

    def buildDataSubset(self):
        target = metadata_df[self.target].astype(int)  
        pos_class = target[target == 1].index
        neg_class = target[target == 0].index
        
        n_pos = len(pos_class)
        n_neg = len(neg_class)
        
        if n_pos > n_neg: 
            temp = pos_class
            pos_class = neg_class
            neg_class = temp
        cm = cosine_similarity(metadata_matching.loc[pos_class, :], metadata_matching.loc[neg_class, :])
        cm = pd.DataFrame(cm, index = pos_class, columns = neg_class)
        cohort_pairs = []
        cohort = []
        for pos_index in cm.index:
            neg_match = cm.loc[pos_index,:].idxmax(axis = 1)
            #print pos_index, "closest sample: ", neg_match
            cm.drop(neg_match, axis = 1, inplace = True)
            cohort.append(pos_index)
            cohort.append(neg_match)
            cohort_pairs.append([pos_index, neg_match])
        n_samples = len(cohort)
        X = otu_df.loc[cohort,:].astype(float).values
        y = target[cohort]
        cohort_pairs = pd.DataFrame(cohort_pairs, columns = ["test", "control"])
        cohort_pairs.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/feature_pairs/" + self.filename + ".csv")
        return X, y, n_samples        

        
    def plotROC(self, tprs, mean_fpr, aucs):
        plt.figure(1, figsize=(6, 6))
        plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle=':', lw=2, color='k', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='black',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=0.9)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3, label=r'$\pm$ 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.title)
        plt.legend(loc="lower right")
        if self.save == True:
            plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/ROCs/" + self.filename + ".png", dpi = 300)
        plt.show()

    def CVHypeSearch(self, X,y):
        print("Hyper Parameter Optimization: 50 model CV.")
        clf_xgb = XGBClassifier(objective = 'binary:logitraw', nthread = 4, seed =124124)
        param_dist = {'n_estimators': [256, 512],
                      "max_depth": [4, 5, 6, 8, 9],
                      "min_child_weight": [1, 3, 5, 7],
                      "gamma": [0.0, 0.1, 0.2 , 0.3],
                      "colsample_bytree": [0.7, 0.8, 1.0]}  
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43422)    
        clf = RandomizedSearchCV(clf_xgb, param_distributions = param_dist, n_iter = 50, scoring = 'roc_auc', error_score = 0, verbose = 0)
        clf.fit(x_train, y_train, eval_metric ='auc', early_stopping_rounds = 15, eval_set=[(x_test, y_test)], verbose = False)
        best_params = clf.best_params_
        print(best_params)
        return best_params

    def RepeatedKFoldCV(self, X, y):
        print("10-repeat 10-fold cv")
        tprs = []                    
        aucs = []                         
        importances = []                  
        mean_fpr = np.linspace(0, 1, 101)  
        alg = XGBClassifier(n_estimators=512, 
                            max_depth=6, 
                            learning_rate=0.1, 
                            colsample_bytree=0.8,
                            gamma=0.1, 
                            nthread=4,
                            min_child_weight=1, 
                            objective='binary:logitraw', 
                            seed=1235)
        #if self.optimize: 
        #    best_params = self.CVHypeSearch(X,y)
        #    alg.set_params(**best_params)
        ##Repeat Cross Validation
        for n in np.random.randint(0, high=10000, size = 10):
            cv = KFold(n_splits = 10, random_state = n, shuffle = True)
            for fold_num, (train, test) in enumerate(cv.split(X, y)): 
                alg.fit(X[train], y[train], verbose=False, eval_set=[(X[test], y[test])], eval_metric='auc', early_stopping_rounds=20) 
                probas_ = alg.predict_proba(X[test])
                y_pred = probas_[:,1]
                y_true = y[test]
                imp = alg.feature_importances_
                importances.append(imp)
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
        importances = np.stack(importances)
        importances = pd.DataFrame(importances, columns = otu_df.columns).mean(axis = 0).sort_values(ascending = False)
        importances.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Importances/" + self.filename + ".csv")
        self.plotROC(tprs, mean_fpr, aucs)
        return aucs
        
    def classifyFeature(self):
        X, y, n_samples = self.buildDataSubset()
        print "Cohort Constructed"    
        aucs = self.RepeatedKFoldCV(X, y)
        return aucs

    
#%%

binary_classifier_info = pd.DataFrame([], index = binary_features, columns = ["aucs","n_samples"])
for feature in binary_features:
    feature_counts = metadata_df[feature].value_counts()
    if 1 not in feature_counts.index:
        print "no positive class"
        continue
    n_feature_positive = feature_counts[1]
    if n_feature_positive < 50:
        print feature, ":", str(n_feature_positive), "positive samples. Not enouch to classify"
        continue
    else:
        print feature, ":", str(n_feature_positive), "positive samples"
        AGP = AGPCohortClassification(feature, False, True, "Classification of " + feature.replace("_", " "), feature)
        feat_aucs = AGP.classifyFeature()
        binary_classifier_info.loc[feature, "aucs"] = feat_aucs  
        binary_classifier_info.loc[feature, "n_samples"] = n_feature_positive * 2
        print 
        print

binary_classifier_info["auc_mean"] = binary_classifier_info["aucs"].apply(np.mean)
binary_classifier_info["auc_std"] = binary_classifier_info["aucs"].apply(np.std)

binary_classifier_info = binary_classifier_info.sort_values("mean_auc", ascending  = False)  
binary_classifier_info = binary_classifier_info[np.isfinite(binary_classifier_info['mean_auc'])]
binary_classifier_info.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Binary_feature_results.csv")

#%%


counts = metadata_df["pku"].value_counts()
if 1 not in counts.index:
    print "no positive class"


#%%
boxplotdata = np.vstack(binary_classifier_info["aucs"].values)
boxplotdata = pd.DataFrame(boxplotdata, index = binary_classifier_info.index).T
plt.figure(figsize = (9, 5))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r")
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.ylabel("AUC")
plt.title("XGBoost Cross Fold Validation Perforances of Binary Host Features")
plt.show()
        

#%%
    
def mapOTU(df, taxa_df, colname):
    ### CONCATENATE TAXONOMY TO GIVEN LEVEL
    taxa_df = taxa_df[taxa_df.index.isin(df.columns)]
    taxa_path = taxa_df["Kingdom"]
    for level in taxa_df.columns[1:-1]:
        taxa_path = taxa_path.str.cat(taxa_df[level], sep='_')
        if level == colname:
            break
    ### GET GROUPS OF OTUS BELLONGING TO COMMON TAXONOMY
    taxa_groups = taxa_path.to_frame(0).groupby([0])
    print colname
    print len(taxa_groups.groups)
    summedOTUs = pd.DataFrame([],columns = taxa_groups.groups)
    ### SUM OTU COUNTS
    for group in taxa_groups.groups:
        otu_list = taxa_groups.groups[group].values
        summedOTUs[group] = df[otu_list].sum(axis = 1)
    return summedOTUs



