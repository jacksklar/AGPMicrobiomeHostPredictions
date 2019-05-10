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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score
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

#otu_df = otu_df.div(10000, axis=0)
#otu_df_full = otu_df
#otu_max_abund =  otu_df.mean(axis = 0).div(10000)
#otu_sig = otu_max_abund[otu_max_abund > 0.0001].index
#otu_df = otu_df.loc[:, otu_sig]

#%%


class modelResults:
    def __init__(self):
        self.tprs = []                    
        self.aucs = []                         
        self.importances = []
        self.accuracy = []                         
        self.precision = []                         
        self.recall = []                         
        self.f1 = []             
        self.shuffled_aucs = [] 
        self.shuffled_f1 = [] 
        self.shuffled_tprs = []   
        self.mean_fpr = np.linspace(0, 1, 101)  
    
    def getMetrics(self, cohort_n):
        metrics = pd.Series([])
        metrics.loc["n_samples"] = cohort_n
        metrics.loc["auc_mean"] = np.mean(self.aucs)
        metrics.loc["auc_std"] = np.std(self.aucs)
        metrics.loc["auc_median"] = np.median(self.aucs)
        metrics.loc["shuffled_auc_mean"] = np.mean(self.shuffled_aucs)
        metrics.loc["shuffled_auc_median"] = np.median(self.shuffled_aucs)
        
        data1 = np.reshape(self.aucs, (20, 5))
        data2 = np.reshape(self.shuffled_aucs, (20, 5))
        t_stat, p_val = stats.ttest_ind(data1,data2, axis = 1)
        metrics.loc["p_val"] = np.mean(p_val)
        metrics.loc["t_stat"] = np.mean(t_stat)
        
        metrics.loc["acc_mean"] = np.mean(self.accuracy)
        metrics.loc["acc_std"] = np.std(self.accuracy)
        metrics.loc["recall_mean"] = np.mean(self.recall)
        metrics.loc["recall_std"] = np.std(self.recall)
        metrics.loc["prec_mean"] = np.mean(self.precision)
        metrics.loc["prec_std"] = np.std(self.precision)
        metrics.loc["f1_mean"] = np.mean(self.f1)
        metrics.loc["f1_std"] = np.std(self.f1)
        metrics.loc["shuffled_f1_mean"] = np.mean(self.shuffled_f1) 
        metrics.loc["shuffled_f1_std"] = np.std(self.shuffled_f1)

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

        ##Repeat Cross Validation
        for n in range(1, 21):
            print(str(n) + ","),
            y_shuffled = shuffle(y)
            cv = KFold(n_splits = 5, shuffle = True)
            for fold_num, (train, test) in enumerate(cv.split(X, y, y_shuffled)):                 

                self.trainXGB(X[train], X[test], y[train], y[test], False)
                self.trainXGB(X[train], X[test], y_shuffled[train], y_shuffled[test], True)
                                
                self.trainRF(X[train], X[test], y[train], y[test], False)
                self.trainRF(X[train], X[test], y_shuffled[train], y_shuffled[test], True)
                
                self.trainLasso(X[train], X[test], y[train], y[test], False)
                self.trainLasso(X[train], X[test], y_shuffled[train], y_shuffled[test], True)
                
        if self.plot:
            self.xgb.plotROC(self.feature_name, self.save, self.title)
    
    
    def trainXGB(self, X_train, X_test, y_train, y_test, shuffle):
        alg = XGBClassifier(n_estimators=256, max_depth=2, learning_rate=0.1, colsample_bytree = 0.7, 
                           nthread=4, scoring='roc_auc', reg_alpha = 10, reg_lambda = 5, subsample = 0.5,
                           objective='binary:logistic', seed= 1235) 
        alg.fit(X_train, y_train, verbose=False, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=20)   
        y_pred = alg.predict_proba(X_test)[:,1]
        y_pred_class = alg.predict(X_test)
        y_true = y_test
        
        ##Evaluation Metrics:
        imp = alg.feature_importances_        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_true, y_pred_class)
        acc = accuracy_score(y_true, y_pred_class)
        prec = precision_score(y_true, y_pred_class)
        rec = recall_score(y_true, y_pred_class)
        
        if shuffle:
            self.xgb.shuffled_tprs.append(interp(self.xgb.mean_fpr, fpr, tpr))  ##For plotting roc curves
            self.xgb.shuffled_tprs[-1][0] = 0.0
            self.xgb.shuffled_aucs.append(roc_auc)
            self.xgb.shuffled_f1.append(f1)  
        else:
            self.xgb.importances.append(imp)
            self.xgb.tprs.append(interp(self.xgb.mean_fpr, fpr, tpr))  ##For plotting roc curves
            self.xgb.tprs[-1][0] = 0.0
            self.xgb.aucs.append(roc_auc)
            self.xgb.accuracy.append(acc)                        
            self.xgb.precision.append(prec)                          
            self.xgb.recall.append(rec)                         
            self.xgb.f1.append(f1)  
    
    def trainRF(self, X_train, X_test, y_train, y_test, shuffle):
        alg = RandomForestClassifier(n_estimators=128, criterion= 'gini', min_samples_split=2, min_samples_leaf=1)
        alg.fit(X_train, y_train)
        y_pred = alg.predict_proba(X_test)[:,1]
        y_pred_class = alg.predict(X_test)
        y_true = y_test
        
        ##Evaluation Metrics:
        imp = alg.feature_importances_        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_true, y_pred_class)
        acc = accuracy_score(y_true, y_pred_class)
        prec = precision_score(y_true, y_pred_class)
        rec = recall_score(y_true, y_pred_class)
        
        if shuffle:
            self.rf.shuffled_tprs.append(interp(self.rf.mean_fpr, fpr, tpr))  ##For plotting roc curves
            self.rf.shuffled_tprs[-1][0] = 0.0
            self.rf.shuffled_aucs.append(roc_auc)
            self.rf.shuffled_f1.append(f1)  
        else:
            self.rf.importances.append(imp)
            self.rf.tprs.append(interp(self.rf.mean_fpr, fpr, tpr))  ##For plotting roc curves
            self.rf.tprs[-1][0] = 0.0
            self.rf.aucs.append(roc_auc)
            self.rf.accuracy.append(acc)                        
            self.rf.precision.append(prec)                          
            self.rf.recall.append(rec)                         
            self.rf.f1.append(f1)  
            
    def trainLasso(self, X_train, X_test, y_train, y_test, shuffle):
        alg = LogisticRegression(solver = 'liblinear', penalty = "l1", class_weight =  'balanced')
        alg.fit(X_train, y_train)
        probas_ = alg.predict_proba(X_test)
        y_pred_class = alg.predict(X_test)
        y_pred = probas_[:,1]
        y_true = y_test
        
        imp = alg.coef_[0,:]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_true, y_pred_class)
        acc = accuracy_score(y_true, y_pred_class)
        prec = precision_score(y_true, y_pred_class)
        rec = recall_score(y_true, y_pred_class)
        
        if shuffle:
            self.lasso.shuffled_tprs.append(interp(self.lasso.mean_fpr, fpr, tpr))  ##For plotting roc curves
            self.lasso.shuffled_tprs[-1][0] = 0.0
            self.lasso.shuffled_aucs.append(roc_auc)
            self.lasso.shuffled_f1.append(f1)  
        else:
            self.lasso.importances.append(imp)
            self.lasso.tprs.append(interp(self.lasso.mean_fpr, fpr, tpr))  ##For plotting roc curves
            self.lasso.tprs[-1][0] = 0.0
            self.lasso.aucs.append(roc_auc)
            self.lasso.accuracy.append(acc)                        
            self.lasso.precision.append(prec)                          
            self.lasso.recall.append(rec)                         
            self.lasso.f1.append(f1)  
        
    

#%%

save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Xgb_genus/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Feature_cohorts/"
feature_list = os.listdir(dir_path)
col_names = otu_df.columns



metrics = ["p_val", "t_stat", "n_samples", "auc_mean", "auc_std", "auc_median", 
           "shuffled_auc_mean",  "shuffled_auc_median", "acc_mean", "acc_std", 
           "recall_mean", "recall_std", "prec_mean", "prec_std", "f1_mean", 
           "f1_std", "shuffled_f1_mean", "shuffled_f1_std"]

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
    
    
    xgb_output = CohClass.xgb    
    xgb_results.loc[feature_name, :] = xgb_output.getMetrics(cohort_n)
    xgb_importances.loc[feature_name, :] = xgb_output.getImportances(col_names)
    xgb_aucs.loc[feature_name, :] = xgb_output.aucs  
    xgb_shuffled_aucs.loc[feature_name, :] = xgb_output.shuffled_aucs

    rf_output = CohClass.rf    
    rf_results.loc[feature_name, :] = rf_output.getMetrics(cohort_n)
    rf_importances.loc[feature_name, :] = rf_output.getImportances(col_names)
    rf_aucs.loc[feature_name, :] = rf_output.aucs  
    rf_shuffled_aucs.loc[feature_name, :] = rf_output.shuffled_aucs
    
    lasso_output = CohClass.lasso    
    lasso_results.loc[feature_name, :] = lasso_output.getMetrics(cohort_n)
    lasso_importances.loc[feature_name, :] = lasso_output.getImportances(col_names)
    lasso_aucs.loc[feature_name, :] = lasso_output.aucs  
    lasso_shuffled_aucs.loc[feature_name, :] = lasso_output.shuffled_aucs
    break

#%%
    
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


#%%

####BINARY BOXPLOTSS 
temp = xgb_results[xgb_results["p_val"] <= 0.05]
boxplotdata = xgb_aucs.loc[temp.index, :].values
boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
plt.figure(figsize = (5, 10))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.ylabel("")
plt.title("Cross-Fold Validation of Host Lifestyle/Disease Variables")
plt.show()


#%%


save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Genus_Frequency/"
#save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/OTU_Frequency/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Frequency_cohorts/"
frequency_list = os.listdir(dir_path)
frequency_results = pd.DataFrame([], columns = ["p_val", "t_stat", "n_samples", "acc_mean", "acc_std", "recall_mean", "recall_std", "prec_mean", "prec_std", "f1_mean", "f1_std", "shuffled_f1_mean", "shuffled_f1_std"])

frequency_aucs = pd.DataFrame([], columns = range(500))
frequency_shuffled_aucs = pd.DataFrame([], columns = range(500))


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
        feature_importance[freq_name] = CohClass.importances
        
        frequency_aucs.loc[feature_name, :] = CohClass.aucs  
        frequency_shuffled_aucs.loc[feature_name, :] = CohClass.shuffled_aucs
    
        frequency_results.loc[feature_name, "auc_mean"] = np.mean(CohClass.aucs)
        frequency_results.loc[feature_name, "auc_std"] = np.std(CohClass.aucs)
        frequency_results.loc[feature_name, "auc_median"] = np.median(CohClass.aucs)
        frequency_results.loc[feature_name, "shuffled_auc_mean"] = np.mean(CohClass.shuffled_aucs)
        frequency_results.loc[feature_name, "shuffled_auc_median"] = np.median(CohClass.shuffled_aucs)
        
        data1 = np.reshape(CohClass.aucs, (100, 5))
        data2 = np.reshape(CohClass.shuffled_aucs, (100, 5))
        t_stat, p_val = stats.ttest_ind(data1,data2, axis = 1)
        #p_val, t_stat = independent_ttest(data1, data2)
        frequency_results.loc[feature_name, "p_val"] = np.mean(p_val)
        frequency_results.loc[feature_name, "t_stat"] = np.mean(t_stat)
        
        frequency_results.loc[feature_name, ["acc_mean", "acc_std"]] = [np.mean(CohClass.accuracy), np.std(CohClass.accuracy)]
        frequency_results.loc[feature_name, ["recall_mean", "recall_std"]] = [np.mean(CohClass.recall), np.std(CohClass.recall)]
        frequency_results.loc[feature_name, ["prec_mean", "prec_std"]] = [np.mean(CohClass.precision), np.std(CohClass.precision)]
        frequency_results.loc[feature_name, ["f1_mean", "f1_std"]] = [np.mean(CohClass.f1), np.std(CohClass.f1)]   
        frequency_results.loc[feature_name, ["shuffled_f1_mean", "shuffled_f1_std"]] = [np.mean(CohClass.shuffled_f1), np.std(CohClass.shuffled_f1)]   
        frequency_results.loc[feature_name, "n_samples"] = len(cohort)
        
        ##MultiROC plot
        mean_tpr = np.mean(CohClass.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(CohClass.mean_fpr, mean_tpr)
        std_auc = np.std(CohClass.aucs)
        plt.plot(CohClass.mean_fpr, 
                 mean_tpr,
                 label=r'%s (AUC=%0.2f$\pm$%0.2f)' % (freq_name.capitalize(), mean_auc, std_auc), 
                 lw=2, 
                 alpha=0.9, 
                 color = c)
        std_tpr = np.std(CohClass.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(CohClass.mean_fpr, tprs_lower, tprs_upper, color = c, alpha=.1,)
        
    feature_importance.to_csv(save_path + "/Importances/" + feature + ".csv")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Classification of " + feature.replace("_", " ").capitalize())
    plt.legend(loc="lower right")
    plt.savefig(save_path + "ROCs/" + feature + ".png", dpi = 300)
    plt.show()

        

frequency_results = frequency_results.sort_values("auc_median", ascending  = False)  
frequency_results.to_csv(save_path + "frequency_classification_results.csv")
frequency_aucs.to_csv(save_path + "frequency_aucs.csv")
frequency_shuffled_aucs.to_csv(save_path + "frequency_shuffled_aucs.csv")

#%%

###FREQUNECY BOXPLOTS
for val in frequency_info["Variable"].unique():
    freq_groups = frequency_info[frequency_info["Variable"] == val].index
    temp = frequency_results.loc[freq_groups,:].sort_values("auc_median", ascending  = False)
    group_names = frequency_info.loc[temp.index, "plot_name"]
    boxplotdata = np.vstack(temp["aucs"].values)
    boxplotdata = pd.DataFrame(boxplotdata, index = group_names).T
    g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
    plt.xlabel("AUC")
    plt.ylabel("")
    plt.title(val.replace("_", " ").capitalize())
    plt.tight_layout()
    plt.savefig(save_path + "BoxPlots/" + val + ".png", dpi = 200)
    plt.show()


#%%
"""

    def RepeatedKFoldCV(self, X, y):
        print("10-Repeat 5-Fold CV")
        print("Repeat: ") 
        tprs = []                    
        aucs = []                         
        importances = []                  
        mean_fpr = np.linspace(0, 1, 101)  
        alg = XGBClassifier(n_estimators=256, max_depth=3, learning_rate=0.2, colsample_bytree = 1.0, 
                            reg_alpha = 5, nthread=4, min_child_weight=1, scoring='roc_auc',
                            objective='binary:logistic', seed=1235) #logitraw
        
        ##Repeat Cross Validation
        for n in np.random.randint(0, high=10000, size = 10):
            print(str(n) + ", "),
            cv = KFold(n_splits = 10, random_state = n, shuffle = True)
            for fold_num, (train, test) in enumerate(cv.split(X, y)): 
                y_suffled = shuffle(y[train])
                
                
                ##Fit True Label Data
                alg.fit(X[train], y[train], verbose=False, eval_set=[(X[test], y[test])], eval_metric='auc', early_stopping_rounds=20) 
                probas_ = alg.predict_proba(X[test])
                y_pred = probas_[:,1]
                y_true = y[test]
                
                ##True label feature importances
                imp = alg.feature_importances_
                importances.append(imp)
                
                ##Shuffled Label Null Hypothesis:

                
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                tprs.append(interp(mean_fpr, fpr, tpr))  ##For plotting roc curves
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                
        ###average importance values over all repeats/folds
        importances = np.stack(importances)
        importances = pd.DataFrame(importances, columns = otu_df.columns).mean(axis = 0).sort_values(ascending = False)
        
        ##save importanes/ plot rocs for binary/diseas features (frequency cohorts are saved seperately)
        if self.save :
            importances.to_csv(save_path + "Importances/" + self.feature_name + ".csv")
        if self.plot:
            self.plotROC(tprs, mean_fpr, aucs)
        return aucs, tprs, mean_fpr, importances
        
def independent_ttest(data1, data2):
    n = 5
    ts = []
    ps = []
    for j in range(20):
        repeat1 = data1[j,:]
        repeat2 = data2[j,:]

        mean1, mean2 = np.mean(repeat1), np.mean(repeat2)
        se1, se2 = sem(repeat1), sem(repeat2)
        sed = np.sqrt(se1**2.0 + se2**2.0)
        t_stat = (mean1 - mean2) / sed
        df = 2*n - 2
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
        ts.append(t_stat)
        ps.append(p)
    p = np.mean(ps)
    t_stat = np.mean(ts)
    return p, t_stat
"""






