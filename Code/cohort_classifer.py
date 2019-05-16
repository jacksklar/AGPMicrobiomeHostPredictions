#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:17:44 2019

@author: sklarjg
"""

import os
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns        
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef
from scipy import interp


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

metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
otu_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Otu_Data.csv", index_col = 0)
feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)
frequency_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/frequency_feature_info.csv", index_col = 0)
taxa_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/taxa_md5.xls", sep = "\t", index_col = 0)
taxa_df = taxa_df[taxa_df.index.isin(otu_df.columns)]
taxa_df = taxa_df.replace(np.nan, 'Unknown', regex=True)
otu_df = otu_df.loc[metadata_df.index, :]

otu_df = mapOTU(otu_df, taxa_df, "Genus")
otu_df = otu_df.reindex(otu_df.mean().sort_values(ascending = False).index, axis=1)

#otu_df_full = otu_df
#otu_max_abund =  otu_df.mean(axis = 0).div(10000)
#otu_sig = otu_max_abund[otu_max_abund > 0.0001].index
#otu_df = otu_df.loc[:, otu_sig]

#%%


def empiricalPVal(statistic, null_dist):
    ###number of shuffled iterations where performance is >= standard iteration performance
    count = len([val for val in null_dist if val >= statistic])
    p_val = (count + 1)/float(len(null_dist) + 1)
    return p_val

class modelResults:
    def __init__(self):
        self.tprs = []                    
        self.aucs = []                         
        self.importances = []
        self.accuracy = []                         
        self.matthews = []           
        self.shuffled_accuracy = []
        self.shuffled_aucs = [] 
        self.shuffled_matthews = [] 
        self.shuffled_tprs = []   
        self.mean_fpr = np.linspace(0, 1, 101)  
    
    def getMetrics(self, cohort_n):
        metrics = pd.Series([])
        metrics.loc["n_samples"] = cohort_n
        metrics.loc["auc_mean"] = np.mean(self.aucs)
        metrics.loc["auc_std"] = np.std(self.aucs)
        metrics.loc["auc_median"] = np.median(self.aucs)
        metrics.loc["shuffled_auc_mean"] = np.mean(self.shuffled_aucs)
        metrics.loc["shuffled_auc_std"] = np.std(self.shuffled_aucs)
        metrics.loc["shuffled_auc_median"] = np.median(self.shuffled_aucs)
        metrics.loc["p_val"] = np.mean([empiricalPVal(stat, self.shuffled_aucs) for stat in self.aucs])
        metrics.loc["acc_mean"] = np.mean(self.accuracy)
        metrics.loc["acc_std"] = np.std(self.accuracy)
        
        metrics.loc["matthews_mean"] = np.mean(self.matthews)
        metrics.loc["matthews_std"] = np.std(self.matthews)
        
        metrics.loc["shuffled_matthews_std"] = np.std(self.shuffled_matthews)
        metrics.loc["shuffled_matthews_mean"] = np.mean(self.shuffled_matthews)
        
        metrics.loc["shuffled_accuracy_mean"] = np.mean(self.shuffled_accuracy) 
        metrics.loc["shuffled_accuracy_std"] = np.std(self.shuffled_accuracy)
        return metrics
    
    def getImportances(self, col_names):
        avg_imps = np.stack(self.importances)
        avg_imps = pd.DataFrame(avg_imps, columns = col_names).mean(axis = 0)
        return avg_imps

    def plotROC(self, feature_name, save, title):
        plt.figure(1, figsize=(6, 6))
        plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle=':', lw=2, color='k', alpha=.8)
        
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        plt.plot(self.mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=0.9)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='b', alpha=.3, label=r'$\pm$ 1 std. dev.')
        
        mean_tpr = np.mean(self.shuffled_tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(self.shuffled_tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.shuffled_aucs)
        plt.plot(self.mean_fpr, mean_tpr, color='r',label=r'Mean Shuffled-ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=0.9)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='r', alpha=.3, label=r'$\pm$ 1 std. dev.')
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        
        if save :
            plt.savefig(save_path + "ROCs/" + feature_name + ".png", dpi = 300)
        plt.show()


class AGPCohortClassification:

    def __init__(self, feature_name, cohort, plot, save, title):
        self.feature_name = feature_name
        self.cohort = cohort
        self.plot = plot
        self.save = save
        self.title = title

    def classifyFeature(self):
        X, y = self.buildDataSubset()
        self.RepeatedKFoldCV(X, y)

    def buildDataSubset(self):
        X = otu_df.loc[self.cohort.index,:].astype(float).values
        y = self.cohort["target"]
        print y.value_counts()
        le = LabelEncoder()
        le.fit(y)
        X = np.log(X + 1.0)
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        X, y = shuffle(X, y)
        if len(X) > 2000:
            X = X[:2000,:]
            y = y[:2000]
        return X, y        

    def RepeatedKFoldCV(self, X, y):
        self.xgb = modelResults()
        self.rf = modelResults()
        self.lasso = modelResults()
        self.gnb = modelResults()

        ##Repeat Cross Validation
        #for n in range(1, 21):
            
            #print(str(n) + ","),
            #y_shuffled = shuffle(y)
            #cv = KFold(n_splits = 5, shuffle = True)
        cv = StratifiedShuffleSplit(n_splits=100, test_size=0.3)
        for fold_num, (train, test) in enumerate(cv.split(X, y)):    
            y_shuffled = shuffle(y)
            print(str(fold_num) + ","),
             
            ##EXTREME GRADIENT BOOSTED TREES:
            self.trainModel(X[train], X[test], y[train], y[test], False, self.xgb)
            self.trainModel(X[train], X[test], y_shuffled[train], y_shuffled[test], True, self.xgb)
            ##RANDOM FOREST:
            self.trainModel(X[train], X[test], y[train], y[test], False, self.rf)
            self.trainModel(X[train], X[test], y_shuffled[train], y_shuffled[test], True, self.rf)
            ##LASSO-LOGISTIC REGRESSION:
            self.trainModel(X[train], X[test], y[train], y[test], False, self.lasso)
            self.trainModel(X[train], X[test], y_shuffled[train], y_shuffled[test], True, self.lasso)
            ##GAUSSIAN NAIVE BAYES:
            self.trainModel(X[train], X[test], y[train], y[test], False, self.gnb)
            self.trainModel(X[train], X[test], y_shuffled[train], y_shuffled[test], True, self.gnb)
            
        if self.plot:
            self.xgb.plotROC(self.feature_name, self.save, self.title)
    
    
    
    def trainModel(self, X_train, X_test, y_train, y_test, shuffle, model):
        
        if model == self.xgb:
            alg = XGBClassifier(n_estimators=256, max_depth=3, learning_rate=0.1, colsample_bytree = 0.5, 
                                nthread=4, reg_alpha = 10, reg_lambda = 5, objective='binary:logistic') 
            alg.fit(X_train, y_train, verbose=False, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=20)   
            imp = alg.feature_importances_   
        if model == self.rf:        
            alg = RandomForestClassifier(n_estimators=128, criterion= 'gini', min_samples_split=2, min_samples_leaf=1)
            alg.fit(X_train, y_train)
            imp = alg.feature_importances_        
        if model == self.lasso:        
            alg = LogisticRegression(solver = 'liblinear', penalty = "l1", class_weight =  'balanced')
            alg.fit(X_train, y_train)
            imp = alg.coef_[0,:]
        if model == self.gnb:    
            alg = GaussianNB()
            alg.fit(X_train, y_train)
            imp = np.zeros(X_train.shape[1])
            
        y_pred = alg.predict_proba(X_test)[:,1]
        y_pred_class = alg.predict(X_test)
        y_true = y_test
        
        ##Evaluation Metrics:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        matthew = matthews_corrcoef(y_true, y_pred_class)
        acc = accuracy_score(y_true, y_pred_class)
        
        if shuffle:
            model.shuffled_tprs.append(interp(self.xgb.mean_fpr, fpr, tpr))  ##For plotting roc curves
            model.shuffled_tprs[-1][0] = 0.0
            model.shuffled_aucs.append(roc_auc)
            model.shuffled_matthews.append(matthew)  
            model.shuffled_accuracy.append(acc)
        else:
            model.importances.append(imp)
            model.tprs.append(interp(model.mean_fpr, fpr, tpr))  ##For plotting roc curves
            model.tprs[-1][0] = 0.0
            model.aucs.append(roc_auc)
            model.accuracy.append(acc)                                               
            model.matthews.append(matthew)      


#%%

save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/genus_binary_results/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/"
feature_list = os.listdir(dir_path)
col_names = otu_df.columns

metrics = ["p_val", "n_samples", "auc_mean", "auc_std", "auc_median", 
           "shuffled_auc_mean", "shuffled_auc_std",  "shuffled_auc_median", "acc_mean", "acc_std", 
           "shuffled_matthews_mean", "shuffled_matthews_std","matthews_mean", "matthews_std",
           "shuffled_accuracy_mean", "shuffled_accuracy_std"]

xgb_results = pd.DataFrame([], columns = metrics)
xgb_importances = pd.DataFrame([], columns = col_names)
xgb_aucs = pd.DataFrame([], columns = range(100))
xgb_shuffled_aucs = pd.DataFrame([], columns = range(100))

rf_results = pd.DataFrame([], columns = metrics)
rf_importances = pd.DataFrame([], columns = col_names)
rf_aucs = pd.DataFrame([], columns = range(100))
rf_shuffled_aucs = pd.DataFrame([], columns = range(100))

lasso_results = pd.DataFrame([], columns = metrics)
lasso_importances = pd.DataFrame([], columns = col_names)
lasso_aucs = pd.DataFrame([], columns = range(100))
lasso_shuffled_aucs = pd.DataFrame([], columns = range(100))

gnb_results = pd.DataFrame([], columns = metrics)
gnb_importances = pd.DataFrame([], columns = col_names)
gnb_aucs = pd.DataFrame([], columns = range(100))
gnb_shuffled_aucs = pd.DataFrame([], columns = range(100))

for feature in feature_list:
    feature_name = feature.split(".")[0]
    print feature_name
    if feature_name == "":
        continue
    if feature_info.loc[feature_name, "type"] == "Race" or feature_info.loc[feature_name, "type"]  == "other":
        continue
    
    cohort = pd.read_csv(dir_path + feature, index_col = 0) 
    cohort_n = len(cohort)
    CohClass = AGPCohortClassification(feature_name, cohort, True, True, "Classification of " + feature_info.loc[feature_name, "plot_name"])
    CohClass.classifyFeature()
    
    xgb_results.loc[feature_name, :] = CohClass.xgb.getMetrics(cohort_n)
    xgb_importances.loc[feature_name, :] = CohClass.xgb.getImportances(col_names)
    xgb_aucs.loc[feature_name, :] = CohClass.xgb.aucs  
    xgb_shuffled_aucs.loc[feature_name, :] = CohClass.xgb.shuffled_aucs

    rf_results.loc[feature_name, :] = CohClass.rf.getMetrics(cohort_n)
    rf_importances.loc[feature_name, :] = CohClass.rf.getImportances(col_names)
    rf_aucs.loc[feature_name, :] = CohClass.rf.aucs  
    rf_shuffled_aucs.loc[feature_name, :] = CohClass.rf.shuffled_aucs
    
    lasso_results.loc[feature_name, :] = CohClass.lasso.getMetrics(cohort_n)
    lasso_importances.loc[feature_name, :] = CohClass.lasso.getImportances(col_names)
    lasso_aucs.loc[feature_name, :] = CohClass.lasso.aucs  
    lasso_shuffled_aucs.loc[feature_name, :] = CohClass.lasso.shuffled_aucs
    
    gnb_results.loc[feature_name, :] = CohClass.gnb.getMetrics(cohort_n)
    gnb_importances.loc[feature_name, :] = CohClass.gnb.getImportances(col_names)
    gnb_aucs.loc[feature_name, :] = CohClass.gnb.aucs  
    gnb_shuffled_aucs.loc[feature_name, :] = CohClass.gnb.shuffled_aucs

xgb_results.to_csv(save_path + "xgb_results.csv")
xgb_aucs.to_csv(save_path + "xgb_aucs.csv")
xgb_shuffled_aucs.to_csv(save_path + "xgb_shuffled_aucs.csv")
xgb_importances.to_csv(save_path + "xgb_importances.csv")

rf_results.to_csv(save_path + "rf_results.csv")
rf_aucs.to_csv(save_path + "rf_aucs.csv")
rf_shuffled_aucs.to_csv(save_path + "rf_shuffled_aucs.csv")
rf_importances.to_csv(save_path + "rf_importances.csv")

lasso_results.to_csv(save_path + "lasso_results.csv")
lasso_aucs.to_csv(save_path + "lasso_aucs.csv")
lasso_shuffled_aucs.to_csv(save_path + "lasso_shuffled_aucs.csv")
lasso_importances.to_csv(save_path + "lasso_importances.csv")

gnb_results.to_csv(save_path + "gnb_results.csv")
gnb_aucs.to_csv(save_path + "gnb_aucs.csv")
gnb_shuffled_aucs.to_csv(save_path + "gnb_shuffled_aucs.csv")
gnb_importances.to_csv(save_path + "gnb_importances.csv")


####BINARY BOXPLOTSS 
temp = xgb_results[xgb_results["p_val"] <= 0.05].sort_values("auc_median", ascending = False)
boxplotdata = xgb_aucs.loc[temp.index, :].values
boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
plt.figure(figsize = (5, 10))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.ylabel("")
plt.show()

temp = rf_results[rf_results["p_val"] <= 0.05].sort_values("auc_median", ascending = False)
boxplotdata = rf_aucs.loc[temp.index, :].values
boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
plt.figure(figsize = (5, 10))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.ylabel("")
plt.show()

temp = lasso_results[lasso_results["p_val"] <= 0.05].sort_values("auc_median", ascending = False)
boxplotdata = lasso_aucs.loc[temp.index, :].values
boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
plt.figure(figsize = (5, 10))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.ylabel("")
plt.show()

#%%


save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/genus_frequency_results/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_cohorts/"
frequency_list = os.listdir(dir_path)

xgb_freq_results = pd.DataFrame([], columns = metrics)
xgb_freq_importances = pd.DataFrame([], columns = col_names)
xgb_freq_aucs = pd.DataFrame([], columns = range(100))
xgb_freq_shuffled_aucs = pd.DataFrame([], columns = range(100))

rf_freq_results = pd.DataFrame([], columns = metrics)
rf_freq_importances = pd.DataFrame([], columns = col_names)
rf_freq_aucs = pd.DataFrame([], columns = range(100))
rf_freq_shuffled_aucs = pd.DataFrame([], columns = range(100))

lasso_freq_results = pd.DataFrame([], columns = metrics)
lasso_freq_importances = pd.DataFrame([], columns = col_names)
lasso_freq_aucs = pd.DataFrame([], columns = range(100))
lasso_freq_shuffled_aucs = pd.DataFrame([], columns = range(100))

gnb_freq_results = pd.DataFrame([], columns = metrics)
gnb_freq_importances = pd.DataFrame([], columns = col_names)
gnb_freq_aucs = pd.DataFrame([], columns = range(100))
gnb_freq_shuffled_aucs = pd.DataFrame([], columns = range(100))


for feature in frequency_info["Variable"].unique():
    cohort_filenames = frequency_info[frequency_info["Variable"] == feature].index.values + ".csv"
    freq_names = [ val.split("_")[-2] for val in cohort_filenames]
    freq_colors = ["b", "y", "r", "g"]
    print feature
    feature_importance = pd.DataFrame([], columns = ['daily', 'regular', 'occasional', 'rare']) 
    plt.figure(1, figsize=(6, 6))
    plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle=':', lw=2, color='k', alpha=.8)

    for filename, freq_name, c in zip(cohort_filenames, freq_names, freq_colors):
        feature_name = filename.split(".")[0]
        print feature_name
        if feature_name == "":
            continue
        cohort = pd.read_csv(dir_path + filename, index_col = 0)    
        CohClass = AGPCohortClassification(feature_name, cohort, False, False, "e")
        CohClass.classifyFeature()

        xgb_freq_results.loc[feature_name, :] = CohClass.xgb.getMetrics(cohort_n)
        xgb_freq_importances.loc[feature_name, :] = CohClass.xgb.getImportances(col_names)
        xgb_freq_aucs.loc[feature_name, :] = CohClass.xgb.aucs  
        xgb_freq_shuffled_aucs.loc[feature_name, :] = CohClass.xgb.shuffled_aucs
        
        rf_freq_results.loc[feature_name, :] = CohClass.rf.getMetrics(cohort_n)
        rf_freq_importances.loc[feature_name, :] = CohClass.rf.getImportances(col_names)
        rf_freq_aucs.loc[feature_name, :] = CohClass.rf.aucs  
        rf_freq_shuffled_aucs.loc[feature_name, :] = CohClass.rf.shuffled_aucs
        
        lasso_freq_results.loc[feature_name, :] = CohClass.lasso.getMetrics(cohort_n)
        lasso_freq_importances.loc[feature_name, :] = CohClass.lasso.getImportances(col_names)
        lasso_freq_aucs.loc[feature_name, :] = CohClass.lasso.aucs  
        lasso_freq_shuffled_aucs.loc[feature_name, :] = CohClass.lasso.shuffled_aucs
        
        gnb_freq_results.loc[feature_name, :] = CohClass.gnb.getMetrics(cohort_n)
        gnb_freq_importances.loc[feature_name, :] = CohClass.gnb.getImportances(col_names)
        gnb_freq_aucs.loc[feature_name, :] = CohClass.gnb.aucs  
        gnb_freq_shuffled_aucs.loc[feature_name, :] = CohClass.gnb.shuffled_aucs
        
        
        ##MultiROC plot
        mean_tpr = np.mean(CohClass.xgb.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(CohClass.xgb.mean_fpr, mean_tpr)
        std_auc = np.std(CohClass.xgb.aucs)
        plt.plot(CohClass.xgb.mean_fpr, 
                 mean_tpr,
                 label=r'%s (AUC=%0.2f$\pm$%0.2f)' % (freq_name.capitalize(), mean_auc, std_auc), 
                 lw=2, 
                 alpha=0.9, 
                 color = c)
        std_tpr = np.std(CohClass.xgb.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(CohClass.xgb.mean_fpr, tprs_lower, tprs_upper, color = c, alpha=.1,)
        
    feature_importance.to_csv(save_path + "/Importances/" + feature + ".csv")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Classification of " + feature.replace("_", " ").capitalize())
    plt.legend(loc="lower right")
    plt.savefig(save_path + "ROCs/" + feature + ".png", dpi = 300)
    plt.show()

        

xgb_freq_results.to_csv(save_path + "xgb_freq_results.csv")
xgb_freq_aucs.to_csv(save_path + "xgb_freq_aucs.csv")
xgb_freq_shuffled_aucs.to_csv(save_path + "xgb_freq_shuffled_aucs.csv")
xgb_freq_importances.to_csv(save_path + "xgb_freq_importances.csv")

rf_freq_results.to_csv(save_path + "rf_freq_results.csv")
rf_freq_aucs.to_csv(save_path + "rf_freq_aucs.csv")
rf_freq_shuffled_aucs.to_csv(save_path + "rf_freq_shuffled_aucs.csv")
rf_freq_importances.to_csv(save_path + "rf_freq_importances.csv")

lasso_freq_results.to_csv(save_path + "lasso_freq_results.csv")
lasso_freq_aucs.to_csv(save_path + "lasso_freq_aucs.csv")
lasso_freq_shuffled_aucs.to_csv(save_path + "lasso_freq_shuffled_aucs.csv")
lasso_freq_importances.to_csv(save_path + "lasso_freq_importances.csv")

gnb_freq_results.to_csv(save_path + "gnb_freq_results.csv")
gnb_freq_aucs.to_csv(save_path + "gnb_freq_aucs.csv")
gnb_freq_shuffled_aucs.to_csv(save_path + "gnb_freq_shuffled_aucs.csv")
gnb_freq_importances.to_csv(save_path + "gnb_freq_importances.csv")



#%%

###FREQUNECY BOXPLOTS
for val in frequency_info["Variable"].unique():
    freq_groups = frequency_info[frequency_info["Variable"] == val].index
    temp = xgb_freq_results.loc[freq_groups,:].sort_values("auc_median", ascending  = False)
    group_names = frequency_info.loc[temp.index, "plot_name"]
    boxplotdata = np.vstack(lasso_aucs.loc[temp.index, :].values)
    boxplotdata = pd.DataFrame(boxplotdata, index = group_names).T
    g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
    plt.xlabel("AUC")
    plt.ylabel("")
    plt.title(val.replace("_", " ").capitalize())
    plt.tight_layout()
    plt.savefig(save_path + "BoxPlots/" + val + ".png", dpi = 200)
    plt.show()


#%%
    
    
    
    
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/age_binary_results/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/age_cohorts/"
feature_list = os.listdir(dir_path)
col_names = otu_df.columns

metrics = ["p_val", "n_samples", "auc_mean", "auc_std", "auc_median", 
           "shuffled_auc_mean", "shuffled_auc_std",  "shuffled_auc_median", "acc_mean", "acc_std", 
           "shuffled_matthews_mean", "shuffled_matthews_std","matthews_mean", "matthews_std",
           "shuffled_accuracy_mean", "shuffled_accuracy_std"]

xgb_results = pd.DataFrame([], columns = metrics)
xgb_importances = pd.DataFrame([], columns = col_names)
xgb_aucs = pd.DataFrame([], columns = range(100))
xgb_shuffled_aucs = pd.DataFrame([], columns = range(100))

rf_results = pd.DataFrame([], columns = metrics)
rf_importances = pd.DataFrame([], columns = col_names)
rf_aucs = pd.DataFrame([], columns = range(100))
rf_shuffled_aucs = pd.DataFrame([], columns = range(100))

lasso_results = pd.DataFrame([], columns = metrics)
lasso_importances = pd.DataFrame([], columns = col_names)
lasso_aucs = pd.DataFrame([], columns = range(100))
lasso_shuffled_aucs = pd.DataFrame([], columns = range(100))

gnb_results = pd.DataFrame([], columns = metrics)
gnb_importances = pd.DataFrame([], columns = col_names)
gnb_aucs = pd.DataFrame([], columns = range(100))
gnb_shuffled_aucs = pd.DataFrame([], columns = range(100))

for feature in feature_list:
    feature_name = feature.split(".")[0]
    print feature_name
    if feature_name == "":
        continue
    cohort = pd.read_csv(dir_path + feature, index_col = 0) 
    cohort_n = len(cohort)
    CohClass = AGPCohortClassification(feature_name, cohort, True, True, "")
    CohClass.classifyFeature()
    
    xgb_results.loc[feature_name, :] = CohClass.xgb.getMetrics(cohort_n)
    xgb_importances.loc[feature_name, :] = CohClass.xgb.getImportances(col_names)
    xgb_aucs.loc[feature_name, :] = CohClass.xgb.aucs  
    xgb_shuffled_aucs.loc[feature_name, :] = CohClass.xgb.shuffled_aucs

    rf_results.loc[feature_name, :] = CohClass.rf.getMetrics(cohort_n)
    rf_importances.loc[feature_name, :] = CohClass.rf.getImportances(col_names)
    rf_aucs.loc[feature_name, :] = CohClass.rf.aucs  
    rf_shuffled_aucs.loc[feature_name, :] = CohClass.rf.shuffled_aucs
    
    lasso_results.loc[feature_name, :] = CohClass.lasso.getMetrics(cohort_n)
    lasso_importances.loc[feature_name, :] = CohClass.lasso.getImportances(col_names)
    lasso_aucs.loc[feature_name, :] = CohClass.lasso.aucs  
    lasso_shuffled_aucs.loc[feature_name, :] = CohClass.lasso.shuffled_aucs
    
    gnb_results.loc[feature_name, :] = CohClass.gnb.getMetrics(cohort_n)
    gnb_importances.loc[feature_name, :] = CohClass.gnb.getImportances(col_names)
    gnb_aucs.loc[feature_name, :] = CohClass.gnb.aucs  
    gnb_shuffled_aucs.loc[feature_name, :] = CohClass.gnb.shuffled_aucs

xgb_results.to_csv(save_path + "xgb_results.csv")
xgb_aucs.to_csv(save_path + "xgb_aucs.csv")
xgb_shuffled_aucs.to_csv(save_path + "xgb_shuffled_aucs.csv")
xgb_importances.to_csv(save_path + "xgb_importances.csv")

rf_results.to_csv(save_path + "rf_results.csv")
rf_aucs.to_csv(save_path + "rf_aucs.csv")
rf_shuffled_aucs.to_csv(save_path + "rf_shuffled_aucs.csv")
rf_importances.to_csv(save_path + "rf_importances.csv")

lasso_results.to_csv(save_path + "lasso_results.csv")
lasso_aucs.to_csv(save_path + "lasso_aucs.csv")
lasso_shuffled_aucs.to_csv(save_path + "lasso_shuffled_aucs.csv")
lasso_importances.to_csv(save_path + "lasso_importances.csv")

gnb_results.to_csv(save_path + "gnb_results.csv")
gnb_aucs.to_csv(save_path + "gnb_aucs.csv")
gnb_shuffled_aucs.to_csv(save_path + "gnb_shuffled_aucs.csv")
gnb_importances.to_csv(save_path + "gnb_importances.csv")


####BINARY BOXPLOTSS 
temp = xgb_results[xgb_results["p_val"] <= 0.05].sort_values("auc_median", ascending = False)
boxplotdata = xgb_aucs.loc[temp.index, :].values
boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
plt.figure(figsize = (5, 10))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.ylabel("")
plt.show()

temp = rf_results[rf_results["p_val"] <= 0.05].sort_values("auc_median", ascending = False)
boxplotdata = rf_aucs.loc[temp.index, :].values
boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
plt.figure(figsize = (5, 10))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.ylabel("")
plt.show()

temp = lasso_results[lasso_results["p_val"] <= 0.05].sort_values("auc_median", ascending = False)
boxplotdata = lasso_aucs.loc[temp.index, :].values
boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
plt.figure(figsize = (5, 10))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.ylabel("")
plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
