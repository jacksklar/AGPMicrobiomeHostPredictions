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
import matplotlib.pyplot as plt
import seaborn as sns        

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef
from scipy import interp

metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
otu_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Otu_Data.csv", index_col = 0)
otu_df = otu_df.loc[metadata_df.index, :]
taxa_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/taxa_md5.xls", sep = "\t", index_col = 0)
taxa_df = taxa_df[taxa_df.index.isin(otu_df.columns)]
taxa_df = taxa_df.replace(np.nan, 'Unknown', regex=True)

## Funtion maps abundance of bacteria from the lowest level (Operational Taxonomuc Unit, OTU) to rank of interest
## Used to test trade-off between loss of information and reduction of feature space
## OTU -> Genus -> Family -> Order -> Class ...
def mapOTU(df, taxa_df, colname):
    taxa_df = taxa_df[taxa_df.index.isin(df.columns)]
    taxa_path = taxa_df["Kingdom"]
    for level in taxa_df.columns[1:-1]:
        taxa_path = taxa_path.str.cat(taxa_df[level], sep='_')
        if level == colname:
            break
    ### Get groups of OTUs belonging to the same phylogenetic tree branch
    taxa_groups = taxa_path.to_frame(0).groupby([0])
    print colname
    print len(taxa_groups.groups)
    summedOTUs = pd.DataFrame([],columns = taxa_groups.groups)
    ### Sum OTU counts
    for group in taxa_groups.groups:
        otu_list = taxa_groups.groups[group].values
        summedOTUs[group] = df[otu_list].sum(axis = 1)
    return summedOTUs

#otu_df = mapOTU(otu_df, taxa_df, "Genus")
#otu_df = otu_df.reindex(otu_df.mean().sort_values(ascending = False).index, axis=1)

##info for plotting questionnaire cohorts
feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)
frequency_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/frequency_feature_info.csv", index_col = 0)

##OTU abundance converted to relative abundance, removal of OTUs with mean relative abundance bellow 0.01% 
otu_df_full = otu_df
otu_max_abund =  otu_df.mean(axis = 0).div(10000)
otu_sig = otu_max_abund[otu_max_abund > 0.0001].index
otu_df = otu_df.loc[:, otu_sig]
print otu_df.shape[1]
num_iterations = 100


RANDOM_STATE_RF = 123123
RANDOM_STATE_CV = 124213

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

    def plotROC(self, feature_name, save, title, model):
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
            plt.savefig(save_path + "ROCs/" + feature_name + "_" + model + ".png", dpi = 300)
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
        self.GroupCV(X, y)

    ## Preprocess questionnaire matched-pair cohort for classification
    ##      - taxonomic relative abundance data is log-transformed with a pseudocount of 1
    ##      - abundance data is not normally distributed so this transformation makes it more suitable for classification
    def buildDataSubset(self):
        X = otu_df.loc[self.cohort.index,:].astype(float).values
        y = self.cohort["target"].astype(float)
        print y.value_counts()
        X = np.log(X + 1.0)
        ## limit cohorts to 2000 samples
        max_samples = 1500
        if len(X) > max_samples:
            X = X[:max_samples,:]
            y = y[:max_samples]
        return X, y        
    
    ## 25 iteration 4-fold cross validation
    ##      - cohort is matched for a set of standard host-variables 
    ##      - pairs are kept grouped between training and test to maintain this balance
    ## 3 standard machine learning classifiers: random forests, ridge-logistic regression, SVM
    ##      - classifiers chosen for performing well with high-dimensional, low sample data that is noisy, and zero-inflated
    ## Target variable shuffled and model trained over same split of data to assess ability for classifier to find signal in noise
    ## Shuffled performance used to obtain significance non-shuffled standard classifiers
    def GroupCV(self, X, y):
        #self.svm = modelResults()
        self.rf = modelResults()
        self.lasso = modelResults()
        ##100 iterations Group Shuffle-Split Cross Validation (matched case-control pairs remain stratified)
        cv = RepeatedStratifiedKFold(n_splits = 4, n_repeats = 25, random_state = RANDOM_STATE_CV) #75/25 training/test split for each iteration
        for fold_num, (train, test) in enumerate(cv.split(X, y)):
            ##standard scale the log-transformed abundance data 
            scaler = StandardScaler().fit(X[train])
            X_train = scaler.transform(X[train])
            X_test = scaler.transform(X[test])
            y_shuffled = shuffle(y)
            print(str(fold_num) + ","),
            ##SVM:
            #self.trainModel(X_train, X_test, y[train], y[test], False, self.svm, alg_svm)
            #self.trainModel(X_train, X_test, y_shuffled[train], y_shuffled[test], True, self.svm, alg_svm)
            ##RANDOM FOREST:
            self.trainModel(X_train, X_test, y[train], y[test], False, self.rf)
            self.trainModel(X_train, X_test, y_shuffled[train], y_shuffled[test], True, self.rf)
            ##RIDGE LOGISTIC REGRESSION:
            self.trainModel(X_train, X_test, y[train], y[test], False, self.lasso)
            self.trainModel(X_train, X_test, y_shuffled[train], y_shuffled[test], True, self.lasso)
        if self.plot:
            self.rf.plotROC(self.feature_name, self.save, self.title, "rf")
            #self.svm.plotROC(self.feature_name, self.save, self.title, "svm")
        print 
        print
    
    ##train one of three classifiers on CV iteration
    ##returns performance metrics, feature importances, saves to classifier object
    def trainModel(self, X_train, X_test, y_train, y_test, shuffle, model_type):
        #if model_type == self.svm:
        #   alg = SVC(C = 0.001, kernel='linear', class_weight = 'balanced', probability=True)  
        #   alg.fit(X_train, y_train)   
        #   imp = alg.coef_[0,:]
        if model_type == self.rf:        
            alg = RandomForestClassifier(n_estimators = 128, min_samples_leaf = 2, n_jobs = -1, class_weight = 'balanced', random_state=RANDOM_STATE_RF)
            alg.fit(X_train, y_train)
            imp = alg.feature_importances_        

        if model_type == self.lasso:        
            alg = LogisticRegression(solver = 'liblinear', penalty = "l2", class_weight =  'balanced')    
            alg.fit(X_train, y_train)
            imp = alg.coef_[0,:]
            
        y_pred = alg.predict_proba(X_test)[:,1]
        y_pred_class = alg.predict(X_test)
        y_true = y_test
        ##Performance Metrics:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        matthew = matthews_corrcoef(y_true, y_pred_class)
        acc = accuracy_score(y_true, y_pred_class)
        
        if shuffle:
            #results for shuffled target variable (corrupted) dataset used as null hypothesis
            model_type.shuffled_tprs.append(interp(self.rf.mean_fpr, fpr, tpr))
            model_type.shuffled_tprs[-1][0] = 0.0
            model_type.shuffled_aucs.append(roc_auc)
            model_type.shuffled_matthews.append(matthew)  
            model_type.shuffled_accuracy.append(acc)
        else:
            #results of classifier on cohort
            model_type.importances.append(imp)
            model_type.tprs.append(interp(model_type.mean_fpr, fpr, tpr)) 
            model_type.tprs[-1][0] = 0.0
            model_type.aucs.append(roc_auc)
            model_type.accuracy.append(acc)                                               
            model_type.matthews.append(matthew)      

## Aggregate classification results of all questionnaire variables into single csv file for analysis/comparison 
class QuestionnaireResults():
    def __init__(self, iters, col_names, model_name, save_path):
        self.iters = iters
        self.col_names = col_names
        self.model_name = model_name
        self.save_path = save_path
        self.model_results = pd.DataFrame([], columns = metrics)
        self.model_importances = pd.DataFrame([], columns = col_names)
        self.model_aucs = pd.DataFrame([], columns = range(iters))
        self.model_shuffled_aucs = pd.DataFrame([], columns = range(iters))
    
    def AppendModelRes(self, model_obj, cohort_n, feature_name):
        self.model_results.loc[feature_name, :] = model_obj.getMetrics(cohort_n)
        self.model_importances.loc[feature_name, :] = model_obj.getImportances(col_names)
        self.model_aucs.loc[feature_name, :] = model_obj.aucs  
        self.model_shuffled_aucs.loc[feature_name, :] = model_obj.shuffled_aucs

    def SaveModelDF(self):
        self.model_results.to_csv(self.save_path + self.model_name + "_results.csv")
        self.model_aucs.to_csv(self.save_path + "AUCs/" + self.model_name + "_aucs.csv")
        self.model_shuffled_aucs.to_csv(self.save_path + "AUCs/" + self.model_name + "_shuffled_aucs.csv")
        self.model_importances.to_csv(self.save_path + "Importances/" + self.model_name + "_importances.csv")

### Plot boxplot of distribution of AUC results from cross-validation for top performing variables
def PlotFeatureBox(model_results, model_aucs, path, model):
    temp = model_results[(model_results["p_val"] <= 0.05) & (model_results["auc_mean"] >= 0.65)].sort_values("auc_median", ascending = False)
    boxplotdata = model_aucs.loc[temp.index, :].values
    boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
    sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
    plt.xlabel("AUC")
    plt.xlim(0.5, 1.0)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path + "auc_dists_" + model + ".pdf")
    plt.show()

metrics = ["p_val", "n_samples", "auc_mean", "auc_std", "auc_median",  "shuffled_auc_mean", "shuffled_auc_std",  "shuffled_auc_median", "acc_mean", "acc_std", 
           "shuffled_matthews_mean", "shuffled_matthews_std","matthews_mean", "matthews_std", "shuffled_accuracy_mean", "shuffled_accuracy_std"]

def PredPipeline(save_path, dir_path):
    feature_list = os.listdir(dir_path)
    col_names = otu_df.columns
    #svm_FR = QuestionnaireResults(num_iterations, col_names, "svm", save_path)
    rf_FR = QuestionnaireResults(num_iterations, col_names, "rf", save_path)
    lasso_FR = QuestionnaireResults(num_iterations, col_names, "lasso", save_path)

    for feature in feature_list:
        feature_name = feature.split(".")[0]
        print feature_name
        if feature_name == "":
            continue
        if feature_name not in feature_info.index.values:
            print "Skipping"
            continue
        cohort = pd.read_csv(dir_path + feature, index_col = 0) 
        cohort_n = len(cohort)
        CohClass = AGPCohortClassification(feature_name, cohort, True, True, "Classification of " + feature_info.loc[feature_name, "plot_name"])
        CohClass.classifyFeature()
        #svm_FR.AppendModelRes(CohClass.svm, cohort_n)
        rf_FR.AppendModelRes(CohClass.rf, cohort_n, feature_name)
        lasso_FR.AppendModelRes(CohClass.lasso, cohort_n, feature_name)

    #svm_FR.SaveModelDF()
    rf_FR.SaveModelDF()
    lasso_FR.SaveModelDF()
    #Plot performance of binary cohorts (disease, lifestyle, etc.)
    #PlotFeatureBox(svm_FR.model_results, svm_FR.model_aucs, save_path, "svm")
    PlotFeatureBox(rf_FR.model_results, rf_FR.model_aucs, save_path, "rf")
    PlotFeatureBox(lasso_FR.model_results, lasso_FR.model_aucs, save_path, "lasso")

def FreqPredPipeline(save_path, dir_path):
    #svm_FR = QuestionnaireResults(num_iterations, col_names, "svm", save_path)
    rf_FR = QuestionnaireResults(num_iterations, col_names, "rf", save_path)
    lasso_FR = QuestionnaireResults(num_iterations, col_names, "lasso", save_path)
    for feature in frequency_list:
        feature_name = feature.split(".")[0]
        print feature_name
        if feature_name == "":
            continue
        if feature_name not in frequency_info.index.values:
            print "Skipping"
            continue
        cohort = pd.read_csv(dir_path + feature, index_col = 0) 
        cohort_n = len(cohort)
        CohClass = AGPCohortClassification(feature_name, cohort, True, True, "Classification of " + " ".join([val.capitalize() for val in feature_name.split("_")[:-3]]) + " " + frequency_info.loc[feature_name, "plot_name"])
        CohClass.classifyFeature()
        #svm_FR.AppendModelRes(CohClass.svm, cohort_n)
        rf_FR.AppendModelRes(CohClass.rf, cohort_n, feature_name)
    lasso_FR.AppendModelRes(CohClass.lasso, cohort_n, feature_name)
    #svm_FR.SaveModelDF()
    rf_FR.SaveModelDF()
    lasso_FR.SaveModelDF()
    #PlotFeatureBox(svm_FR.model_results, svm_FR.model_aucs, save_path, "svm")
    #PlotFeatureBox(rf_FR.model_results, rf_FR.model_aucs, save_path, "rf")
    #PlotFeatureBox(lasso_FR.model_results, lasso_FR.model_aucs, save_path, "lasso")
    
#%%

col_names = otu_df.columns
    
#excluded Groups
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/agp_excluded_results/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/agp_excluded_cohorts/"
feature_list = os.listdir(dir_path)
PredPipeline(save_path, dir_path)

#Binary Questionnaire Variable Cohort Classification
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_no_matching/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/binary_cohorts_no_matching/"
feature_list = os.listdir(dir_path)
PredPipeline(save_path, dir_path)

save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_standard/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/binary_cohorts_standard/"
feature_list = os.listdir(dir_path)
PredPipeline(save_path, dir_path)

##Frequency Groups compared to cohort of "never" participants
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_no_matching/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/frequency_cohorts_no_matching/"
frequency_list = os.listdir(dir_path)
FreqPredPipeline(save_path, dir_path)

##Frequency Groups compared to cohort of "never" participants
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_standard/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/frequency_cohorts_standard/"
frequency_list = os.listdir(dir_path)
FreqPredPipeline(save_path, dir_path)

save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Diabetes_test_results/"
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/Diabetes_test_cohorts/"
feature_list = os.listdir(dir_path)
PredPipeline(save_path, dir_path)



