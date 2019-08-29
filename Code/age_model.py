#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:34:50 2019

@author: sklarjg
"""

import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns        
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RepeatedStratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as f1

metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
otu_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Otu_Data.csv", index_col = 0)
feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)
frequency_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/frequency_feature_info.csv", index_col = 0)
taxa_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/taxa_md5.xls", sep = "\t", index_col = 0)
taxa_df = taxa_df[taxa_df.index.isin(otu_df.columns)]
taxa_df = taxa_df.replace(np.nan, 'Unknown', regex=True)
otu_df = otu_df.loc[metadata_df.index, :]

otu_df_full = otu_df
otu_max_abund =  otu_df.mean(axis = 0).div(10000)
otu_sig = otu_max_abund[otu_max_abund > 0.0001].index
otu_df = otu_df.loc[:, otu_sig]
print otu_df.shape[1]
num_iterations = 100

dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/Age_cohorts/"
age_cohort = pd.read_csv(dir_path + "agp_healthy_cohort_all_ages.csv", index_col = 0)


metadata_df = metadata_df.loc[age_cohort.index, :]
otu_df = otu_df.loc[age_cohort.index, :]


age_range = 5
age_ranges = ["[{0} - {1})".format(age, age + age_range) for age in range(0, 96, age_range)]
age_cohort['age_groups'] = pd.cut(x=age_cohort["age_years"], bins=len(age_ranges), labels=age_ranges)
age_cohort["age_groups"] = age_cohort["age_groups"].cat.add_categories("[75 - 100)")
older_ages = ["[75 - 80)", "[80 - 85)","[85 - 90)","[90 - 95)","[95 - 100)"]
age_cohort.loc[age_cohort['age_groups'].isin(older_ages), "age_groups"] = "[75 - 100)"
age_cohort["age_groups"] = age_cohort["age_groups"].cat.remove_categories(["[95 - 100)","[80 - 85)","[85 - 90)","[90 - 95)", "[75 - 80)"])
print age_cohort["age_groups"].value_counts()

target_map = {"[0 - 5)": 0,
              "[5 - 10)": 1,
              "[10 - 15)": 2,
              "[15 - 20)": 3,
              "[20 - 25)": 4,
              "[25 - 30)": 5,
              "[30 - 35)": 6,
              "[35 - 40)": 7,
              "[40 - 45)": 8,
              "[45 - 50)": 9,
              "[50 - 55)": 10,
              "[55 - 60)": 11,
              "[60 - 65)": 12,
              "[65 - 70)": 13,
              "[70 - 75)": 14,
              "[75 - 100)": 15}
age_cohort["target"] = age_cohort["age_groups"].map(target_map)

def BalanceClasses(input_df, output_df, target, max_class, min_class):
    unique_classes = output_df[target].unique()
    resampled_ids = pd.Index([])
    for group in unique_classes:
        class_cohort = output_df[output_df[target] == group].index  
        ##Up-Samp
        if len(class_cohort) < min_class:    
            class_cohort = output_df[output_df[target] == group].sample(min_class, replace = True).index
            resampled_ids = resampled_ids.union(class_cohort)
        if len(class_cohort) > max_class: 
            class_cohort = output_df[output_df[target] == group].sample(max_class).index
            resampled_ids = resampled_ids.union(class_cohort)
        else:
            resampled_ids = resampled_ids.union(class_cohort)
            
    output_df = output_df.loc[resampled_ids,:]
    input_df = input_df.loc[resampled_ids,:] 
    return input_df, output_df


#391 = median class count before downsampling 
#otu_df, age_cohort = BalanceClasses(otu_df, age_cohort, "age_groups", 650, 10)
#age_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/Age_cohorts/age_group_cohort.csv") 

us_age_cohort = age_cohort.loc[age_cohort["country"].isin(["USA", "Canada"]), :]
uk_age_cohort = age_cohort.loc[age_cohort["country"] == "United Kingdom", :]

age_count = age_cohort["age_groups"].value_counts(sort = False)
us_age_count = us_age_cohort["age_groups"].value_counts(sort = False)
uk_age_count = uk_age_cohort["age_groups"].value_counts(sort = False)
ratio = us_age_count / age_count

age_compare = pd.DataFrame({"us_counts": us_age_count,"uk_counts": uk_age_count,"full_counts": age_count, "us_ratio": ratio})

age_compare["new_full_counts"] = age_compare["us_counts"]* 1.25
age_compare["new_full_counts"] = age_compare["new_full_counts"].astype(int)


age_compare["delta"] =  age_compare["full_counts"] - age_compare["new_full_counts"]
age_compare["new_uk"] = age_compare["uk_counts"] - age_compare["delta"] 
age_compare["new_us_ratio"] = age_compare["us_counts"]/age_compare["new_full_counts"]


for group in age_compare.index:
    group_info = age_compare.loc[group, :]
    delta = int(group_info["delta"])
    print group, delta

    if delta > 0:    
        print "removing from UK"
        temp_inds = uk_age_cohort[uk_age_cohort["age_groups"] == group].index.values
        temp_inds = temp_inds[:delta]
        age_cohort.drop(temp_inds, axis = 0, inplace = True)
    else:
        print "removing from USA"
        temp_inds = us_age_cohort[us_age_cohort["age_groups"] == group].index.values
        delta = -1 * delta
        temp_inds = temp_inds[: delta]
        age_cohort.drop(temp_inds, axis = 0, inplace = True)

age_cohort.sort_values(by = ["target"], inplace = True)
age_counts = age_cohort["target"].value_counts(sort = False)
plt.bar(age_counts.index, age_counts.values)
plt.show()


#age_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/Age_cohorts/age_group_cohort_balanced.csv") 
print age_counts.median()


otu_df, age_cohort = BalanceClasses(otu_df, age_cohort, "age_groups", 350, 10)
#age_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/Age_cohorts/age_group_cohort_balanced_cutoff.csv") 

#%%
RANDOM_STATE_SVM = 123123
RANDOM_STATE_CV = 124213


group_labels = ["[0 - 5)","[5 - 10)", "[10 - 15)", "[15 - 20)", "[20 - 25)", "[25 - 30)",
                "[30 - 35)", "[35 - 40)", "[40 - 45)", "[45 - 50)", "[50 - 55)", "[55 - 60)",
                "[60 - 65)", "[65 - 70)", "[70 - 75)", "75+"]


X = otu_df.loc[age_cohort.index,:].astype(float).values
y = age_cohort["target"].astype(float)
print y.value_counts()
X = np.log(X + 1.0)
X, y = shuffle(X, y)

cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10, random_state = RANDOM_STATE_CV)

results = []
C_dist = [0.001, 0.01, 0.1, 1.0, 5.0]
C_dist = [0.001]
confusion_mats = []


for c in C_dist:
    print c
    alg_rf = SVC(C = c, kernel='linear', class_weight = 'balanced', random_state = RANDOM_STATE_SVM)  
    f_micros = []
    f_macros = []
    cm_total = pd.DataFrame(0, index=range(16),columns=range(16))
    for fold_num, (train, test) in enumerate(cv.split(X, y)):    
        print(str(fold_num) + ","),
        
        scaler = StandardScaler().fit(X[train])
        X_train = scaler.transform(X[train])
        X_test = scaler.transform(X[test])
        
        y_shuffled = shuffle(y)
        alg_rf.fit(X_train, y[train])
        y_pred = alg_rf.predict(X_test)
        y_true = y[test]
        
        cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
        cm.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/multiclass/Age_cohort/CMs/cm_fold_" + str(fold_num) + ".csv")
        cm_total += cm
        f1_micro = f1(y_true, y_pred, average = 'micro')
        f1_macro = f1(y_true, y_pred, average = 'macro')
        f_micros.append(f1_micro)
        f_macros.append(f1_macro)
        
    micro_avg = np.mean(f_micros)
    macro_avg = np.mean(f_macros)

    print("F1 (Micro): ", micro_avg)
    print("F1 (Macro): ", macro_avg)
    results.append([c, micro_avg, macro_avg])
    
    cm_total = cm_total.astype(float)
    cm_total = cm_total.divide(cm_total.sum(axis=1), axis = 0)   
    cm_total.index = group_labels
    cm_total.columns = group_labels   
    cm_total.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/multiclass/Age_cohort/multiclass_average_conf_mat.csv")

    plt.figure(figsize = (15,12))
    sns.heatmap(cm_total, annot = True, cmap="YlGnBu")
    plt.ylabel("True age group")
    plt.xlabel("Predicted age group")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/multiclass/Age_cohort/Age_svm_c" + str(c) + "_linear.pdf")
    plt.show()
    print 
    
plt.figure(figsize = (20,20))
sns.clustermap(cm_total, annot = True, cmap="YlGnBu", row_cluster=True, col_cluster=False,  method="ward", metric = "cosine", annot_kws={"size": 8})
plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/multiclass/Age_cohort/Age_svm_c_001_linear_clustered_ward.pdf")
plt.show()
    
    
#%%
multiclasses = ["census_region",
                "collection_season",
                "bmi_cat",
                "sleep_duration"]
RANDOM_STATE_SVM = 123123
RANDOM_STATE_CV = 124213
                
                
results = []
for multic in multiclasses:
    curr_var_cohort = metadata_df[~metadata_df[multic].isin(["Not provided", "Unspecified", "Not sure", 5, "Other", "I don't know, I do not have a point of reference"])]
    curr_var_cohort.dropna(subset = [multic], inplace = True)
    
    otu_cohort = otu_df_full.loc[curr_var_cohort.index, :]
    
    class_counts = curr_var_cohort[multic].value_counts()  
    max_class_size = np.median(class_counts.values)
    otu_cohort , meta_cohort = BalanceClasses(otu_cohort, curr_var_cohort, multic, int(max_class_size), 10)
    
    num_classes = len(meta_cohort[multic].unique())
    if type(class_counts.index.values[0]) == str:
        le = LabelEncoder()
        le.fit(meta_cohort[multic].unique())
        y = le.transform(meta_cohort[multic])
    else:
        print "ALERT"
        y = meta_cohort[multic]
    
    for val, val_trans in zip(le.classes_, le.transform(le.classes_)):
        print val + ": " + str(val_trans)

    X = np.log(otu_cohort + 1.0)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    X, y = shuffle(X, y)
        
    #le.inverse_transform
    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10, random_state = RANDOM_STATE_CV)

    alg_rf = SVC(C = 0.01, kernel='linear', class_weight = 'balanced', random_state = RANDOM_STATE_SVM)  
    f_micros = []
    cm_total = pd.DataFrame(0, index=range(num_classes),columns=range(num_classes))
    for fold_num, (train, test) in enumerate(cv.split(X, y)):    
       
        y_shuffled = shuffle(y)
        print(str(fold_num) + ","),
        alg_rf.fit(X[train], y[train])
        #imp = alg_rf.feature_importances_        
        y_pred = alg_rf.predict(X[test])
        y_true = y[test]
        cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
        cm_total += cm
        f1_micro = f1(y_true, y_pred, average = 'micro')
        f_micros.append(f1_micro)
        
    micro_avg = np.mean(f_micros)
    print("F1 (Micro): ", micro_avg)
    results.append([multic, micro_avg])
    
    cm_total = cm_total.astype(float)
    cm_total = cm_total.divide(cm_total.sum(axis=1), axis = 0)   
    cm_total.index = le.classes_
    cm_total.columns = le.classes_    
    cm_total.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/multiclass/" + multic + "_confusion_matrix.csv")
    plt.figure(figsize = (15,12))
    sns.heatmap(cm_total, annot = True, cmap="YlGnBu")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/multiclass/" + multic + "_linear.pdf")
    plt.show()
    print     
    print


