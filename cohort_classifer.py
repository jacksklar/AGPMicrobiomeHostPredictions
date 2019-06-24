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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV
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

#otu_df = mapOTU(otu_df, taxa_df, "Genus")
#otu_df = otu_df.reindex(otu_df.mean().sort_values(ascending = False).index, axis=1)
otu_df_full = otu_df
otu_max_abund =  otu_df.mean(axis = 0).div(10000)
otu_sig = otu_max_abund[otu_max_abund > 0.0001].index
otu_df = otu_df.loc[:, otu_sig]
print otu_df.shape[1]
num_iterations = 100

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
        ##TEST ROC CURVE
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        plt.plot(self.mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=0.9)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='b', alpha=.3, label=r'$\pm$ 1 std. dev.')
        ##SHUFFLED ROC CURVE
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
        y = self.cohort["target"].astype(float)
        print y.value_counts()
        #le = LabelEncoder()
        #le.fit(y)
        X = np.log(X + 1.0)
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        X, y = shuffle(X, y)
        if len(X) > 2000:
            X = X[:2000,:]
            y = y[:2000]
        return X, y        
    
    def HyperparameterOpt(self, X, y):
        print "Hyperparameter Optimization: 100 iterations of 4-fold cv random search"
        parameters_rf = {'max_depth': [40, 60, 80, 100, None],
                         'min_samples_leaf': [1, 2],
                         'n_estimators': [100, 200, 300, 400, 500]}
        rf  = RandomForestClassifier(class_weight = 'balanced')
        rf_random = RandomizedSearchCV(estimator = rf, 
                                       param_distributions = parameters_rf, 
                                       n_iter = 50, 
                                       cv = 3, 
                                       verbose=0, 
                                       random_state=1231242, 
                                       n_jobs = -1)
        rf_random.fit(X, y)
        opt_params = rf_random.best_params_
        print opt_params
        return opt_params

    def RepeatedKFoldCV(self, X, y):
        self.xgb = modelResults()
        self.rf = modelResults()
        self.lasso = modelResults()

        alg_rf = RandomForestClassifier(n_estimators = 128, min_samples_leaf = 2, 
                                        n_jobs = -1, class_weight = 'balanced')
        alg_xgb = XGBClassifier(n_estimators=256, max_depth=3, learning_rate=0.1, 
                            colsample_bytree = 0.7, subsample = 0.7,
                            n_jobs = 4, reg_lambda = 10, objective='binary:logistic') 
        alg_lr = LogisticRegression(solver = 'liblinear', penalty = "l1", class_weight =  'balanced')

        ##Repeat Cross Validation
        cv = StratifiedShuffleSplit(n_splits=num_iterations, test_size=0.3)
        for fold_num, (train, test) in enumerate(cv.split(X, y)):    
            y_shuffled = shuffle(y)
            print(str(fold_num) + ","),
            ##EXTREME GRADIENT BOOSTED TREES:
            self.trainModel(X[train], X[test], y[train], y[test], False, self.xgb, alg_xgb)
            self.trainModel(X[train], X[test], y_shuffled[train], y_shuffled[test], True, self.xgb, alg_xgb)
            ##RANDOM FOREST:
            self.trainModel(X[train], X[test], y[train], y[test], False, self.rf, alg_rf)
            self.trainModel(X[train], X[test], y_shuffled[train], y_shuffled[test], True, self.rf, alg_rf)
            ##LASSO-LOGISTIC REGRESSION:
            self.trainModel(X[train], X[test], y[train], y[test], False, self.lasso, alg_lr)
            self.trainModel(X[train], X[test], y_shuffled[train], y_shuffled[test], True, self.lasso, alg_lr)
        if self.plot:
            self.rf.plotROC(self.feature_name, self.save, self.title)
        print 
        print
    
    def trainModel(self, X_train, X_test, y_train, y_test, shuffle, model_type, alg):
        if model_type == self.xgb:
            alg.fit(X_train, y_train)   
            imp = alg.feature_importances_   
        if model_type == self.rf:        
            alg.fit(X_train, y_train)
            imp = alg.feature_importances_        
        if model_type == self.lasso:        
            alg.fit(X_train, y_train)
            imp = alg.coef_[0,:]
            
        y_pred = alg.predict_proba(X_test)[:,1]
        y_pred_class = alg.predict(X_test)
        y_true = y_test
        ##Evaluation Metrics:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        matthew = matthews_corrcoef(y_true, y_pred_class)
        acc = accuracy_score(y_true, y_pred_class)
        if shuffle:
            model_type.shuffled_tprs.append(interp(self.xgb.mean_fpr, fpr, tpr))  ##For plotting roc curves
            model_type.shuffled_tprs[-1][0] = 0.0
            model_type.shuffled_aucs.append(roc_auc)
            model_type.shuffled_matthews.append(matthew)  
            model_type.shuffled_accuracy.append(acc)
        else:
            model_type.importances.append(imp)
            model_type.tprs.append(interp(model_type.mean_fpr, fpr, tpr))  ##For plotting roc curves
            model_type.tprs[-1][0] = 0.0
            model_type.aucs.append(roc_auc)
            model_type.accuracy.append(acc)                                               
            model_type.matthews.append(matthew)      

class FeatureResults():
    def __init__(self, iters, col_names, model_name, save_path):
        self.iters = iters
        self.col_names = col_names
        self.model_name = model_name
        self.save_path = save_path
        self.model_results = pd.DataFrame([], columns = metrics)
        self.model_importances = pd.DataFrame([], columns = col_names)
        self.model_aucs = pd.DataFrame([], columns = range(iters))
        self.model_shuffled_aucs = pd.DataFrame([], columns = range(iters))
    
    def AppendModelIter(self, model_obj, cohort_n):
        self.model_results.loc[feature_name, :] = model_obj.getMetrics(cohort_n)
        self.model_importances.loc[feature_name, :] = model_obj.getImportances(col_names)
        self.model_aucs.loc[feature_name, :] = model_obj.aucs  
        self.model_shuffled_aucs.loc[feature_name, :] = model_obj.shuffled_aucs

    def SaveModelDF(self):
        self.model_results.to_csv(self.save_path + self.model_name + "_results.csv")
        self.model_aucs.to_csv(self.save_path + "AUCs/" + self.model_name + "_aucs.csv")
        self.model_shuffled_aucs.to_csv(self.save_path + "AUCs/" + self.model_name + "_shuffled_aucs.csv")
        self.model_importances.to_csv(self.save_path + "Importances/" + self.model_name + "_importances.csv")
    
def PlotFeatureBox(model_results, model_aucs):
    temp = model_results[model_results["p_val"] <= 0.05].sort_values("auc_median", ascending = False)
    boxplotdata = model_aucs.loc[temp.index, :].values
    boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
    plt.figure(figsize = (5, 10))
    sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
    plt.xlabel("AUC")
    plt.xlim(0.5, 1.0)
    plt.ylabel("")
    plt.show()

metrics = ["p_val", "n_samples", "auc_mean", "auc_std", "auc_median", 
           "shuffled_auc_mean", "shuffled_auc_std",  "shuffled_auc_median", "acc_mean", "acc_std", 
           "shuffled_matthews_mean", "shuffled_matthews_std","matthews_mean", "matthews_std",
           "shuffled_accuracy_mean", "shuffled_accuracy_std"]

#%%

#Binary Groups
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_updated/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/binary_cohorts_updated/"
feature_list = os.listdir(dir_path)
col_names = otu_df.columns

xgb_FR = FeatureResults(num_iterations, col_names, "xgb", save_path)
rf_FR = FeatureResults(num_iterations, col_names, "rf", save_path)
lasso_FR = FeatureResults(num_iterations, col_names, "lasso", save_path)

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
    xgb_FR.AppendModelIter(CohClass.xgb, cohort_n)
    rf_FR.AppendModelIter(CohClass.rf, cohort_n)
    lasso_FR.AppendModelIter(CohClass.lasso, cohort_n)
xgb_FR.SaveModelDF()
rf_FR.SaveModelDF()
lasso_FR.SaveModelDF()
PlotFeatureBox(xgb_FR.model_results, xgb_FR.model_aucs)
PlotFeatureBox(rf_FR.model_results, rf_FR.model_aucs)
PlotFeatureBox(lasso_FR.model_results, lasso_FR.model_aucs)

##Frequency Groups compared to cohort of "never" participants
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_updated/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/frequency_cohorts_updated/"
frequency_list = os.listdir(dir_path)
xgb_FR = FeatureResults(num_iterations, col_names, "xgb", save_path)
rf_FR = FeatureResults(num_iterations, col_names, "rf", save_path)
lasso_FR = FeatureResults(num_iterations, col_names, "lasso", save_path)

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
        cohort_n = len(cohort)

        xgb_FR.AppendModelIter(CohClass.xgb, cohort_n)
        rf_FR.AppendModelIter(CohClass.rf, cohort_n)
        lasso_FR.AppendModelIter(CohClass.lasso, cohort_n)
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
    
xgb_FR.SaveModelDF()
rf_FR.SaveModelDF()
lasso_FR.SaveModelDF()



#%%
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/age_results/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/age_cohorts/"
feature_list = os.listdir(dir_path)
col_names = otu_df.columns

xgb_FR = FeatureResults(num_iterations, col_names, "xgb", save_path)
rf_FR = FeatureResults(num_iterations, col_names, "rf", save_path)
lasso_FR = FeatureResults(num_iterations, col_names, "lasso", save_path)

for feature in feature_list:
    feature_name = feature.split(".")[0]
    print feature_name
    if feature_name == "":
        continue
    cohort = pd.read_csv(dir_path + feature, index_col = 0) 
    cohort_n = len(cohort)
    CohClass = AGPCohortClassification(feature_name, cohort, True, True, "")
    CohClass.classifyFeature()
    xgb_FR.AppendModelIter(CohClass.xgb, cohort_n)
    rf_FR.AppendModelIter(CohClass.rf, cohort_n)
    lasso_FR.AppendModelIter(CohClass.lasso, cohort_n)

xgb_FR.SaveModelDF()
rf_FR.SaveModelDF()
lasso_FR.SaveModelDF()


rf_results = rf_FR.model_results
rf_results.loc[:, "order"] = [int(val.split("r")[1]) for val in rf_results.index.values]
rf_aucs = rf_FR.model_aucs
temp = rf_results[rf_results["p_val"] <= 0.05].sort_values("order",  ascending = True)
boxplotdata = rf_aucs.loc[temp.index, :].values
labels = [val.split("r")[1] for val in temp.index.values]
boxplotdata = pd.DataFrame(boxplotdata, index = labels).T
plt.figure(figsize = (8, 4))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r")
plt.xlabel("AUC")
plt.ylabel("")
plt.show()


xgb_results = xgb_FR.model_results
xgb_results.loc[:, "order"] = [int(val.split("r")[1]) for val in xgb_results.index.values]
xgb_aucs = xgb_FR.model_aucs
temp = xgb_results[xgb_results["p_val"] <= 0.05].sort_values("order",  ascending = True)
boxplotdata = xgb_aucs.loc[temp.index, :].values
labels = [val.split("r")[1] for val in temp.index.values]
boxplotdata = pd.DataFrame(boxplotdata, index = labels).T
plt.figure(figsize = (8, 4))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r")
plt.xlabel("AUC")
plt.ylabel("")
plt.show()


lasso_results = lasso_FR.model_results
lasso_results.loc[:, "order"] = [int(val.split("r")[1]) for val in lasso_results.index.values]
lasso_aucs = lasso_FR.model_aucs
temp = lasso_results[lasso_results["p_val"] <= 0.05].sort_values("order",  ascending = True)
boxplotdata = lasso_aucs.loc[temp.index, :].values
labels = [val.split("r")[1] for val in temp.index.values]
boxplotdata = pd.DataFrame(boxplotdata, index = labels).T
plt.figure(figsize = (8, 4))
g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r")
plt.xlabel("AUC")
plt.ylabel("")
plt.show()
    

#%%
'''
save_path = "/Users/sklarjg/Desktop/temp/"
updated_cohort_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/binary_cohorts/"
col_names = otu_df.columns


nplant_6_10_cohort = pd.read_csv(updated_cohort_path + "nplant_6_10_cohort.csv", index_col = 0)
nplant_11_20_cohort = pd.read_csv(updated_cohort_path + "nplant_11_20_cohort.csv", index_col = 0)
nplant_21_30_cohort = pd.read_csv(updated_cohort_path + "nplant_21_30_cohort.csv", index_col = 0)
nplant_30_plus_cohort = pd.read_csv(updated_cohort_path + "nplant_30_plus_cohort.csv", index_col = 0)

cohorts = [nplant_6_10_cohort, nplant_11_20_cohort, nplant_21_30_cohort, nplant_30_plus_cohort]
cohort_names = ["nplant_6_10_cohort", "nplant_11_20_cohort", "nplant_21_30_cohort", "nplant_30_plus_cohort"]
plot_names = ["Consumes 6-10 plants", "Consumes 11-20 plants", "Consumes 21-30 plants", "Consumes 30+ plants"]

xgb_FR = FeatureResults(num_iterations, col_names, "xgb", save_path)
rf_FR = FeatureResults(num_iterations, col_names, "rf", save_path)
lasso_FR = FeatureResults(num_iterations, col_names, "lasso", save_path)


for cohort, feature_name, plot_name in zip(cohorts, cohort_names, plot_names):

    CohClass = AGPCohortClassification(feature_name, cohort, True, True, "Classification of " + plot_name)
    CohClass.classifyFeature()
    cohort_n = len(cohort)
    xgb_FR.AppendModelIter(CohClass.xgb, cohort_n)
    rf_FR.AppendModelIter(CohClass.rf, cohort_n)
    lasso_FR.AppendModelIter(CohClass.lasso, cohort_n)
    
xgb_FR.SaveModelDF()
rf_FR.SaveModelDF()
lasso_FR.SaveModelDF()
'''