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
from sklearn.svm import SVC, SVR
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

dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/binary_cohorts_updated/"
age_cohort = pd.read_csv(dir_path + "age_healthy_cohort.csv", index_col = 0)

metadata_df = metadata_df.loc[metadata_df.index.isin(age_cohort.index), :]
otu_df = otu_df.loc[otu_df.index.isin(age_cohort.index), :]


age_range = 5
age_ranges = ["[{0} - {1})".format(age, age + age_range) for age in range(0, 96, age_range)]
age_cohort['age_groups'] = pd.cut(x=age_cohort["age_years"], bins=len(age_ranges), labels=age_ranges)
age_cohort["age_groups"] = age_cohort["age_groups"].cat.add_categories("[80 - 100)")
older_ages = ["[80 - 85)","[85 - 90)","[90 - 95)","[95 - 100)"]
age_cohort.loc[age_cohort['age_groups'].isin(older_ages), "age_groups"] = "[80 - 100)"
age_cohort["age_groups"] = age_cohort["age_groups"].cat.remove_categories(["[95 - 100)","[80 - 85)","[85 - 90)","[90 - 95)"])
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
              "[75 - 80)": 15,
              "[80 - 100)": 16}
    
age_cohort["target"] = age_cohort["age_groups"].map(target_map)
print age_cohort["age_groups"].value_counts()

def BalanceClasses(input_df, output_df, target, max_class, min_class):
    unique_classes = output_df[target].unique()
    print "Unique Classes: ", unique_classes
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
    print output_df[target].value_counts()
    return input_df, output_df

#otu_df, age_cohort = BalanceClasses(otu_df, age_cohort, "age_groups", 300, 10)

#%%

group_labels = ["[0 - 5)","[5 - 10)", "[10 - 15)", "[15 - 20)", "[20 - 25)", "[25 - 30)",
                "[30 - 35)", "[35 - 40)", "[40 - 45)", "[45 - 50)", "[50 - 55)", "[55 - 60)",
                "[60 - 65)", "[65 - 70)", "[70 - 75)", "[75 - 80)", "80+"]


X = otu_df.loc[age_cohort.index,:].astype(float).values
y = age_cohort["target"].astype(float)
print y.value_counts()
#le = LabelEncoder()
#le.fit(y)
X = np.log(X + 1.0)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X, y = shuffle(X, y)

#cv = StratifiedShuffleSplit(n_splits=25, test_size=0.15)
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 12345)

results = []
C_dist = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
C_dist = [0.01]

for c in C_dist:
    print c
    alg_rf = SVC(C = c, kernel='linear', class_weight = 'balanced')  
    f_micros = []
    f_macros = []
    cm_total = pd.DataFrame(0, index=range(17),columns=range(17))
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
    plt.figure(figsize = (15,12))
    sns.heatmap(cm_total, annot = True, cmap="YlGnBu")
    plt.ylabel("True age group")
    plt.xlabel("Predicted age group")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig("/Users/sklarjg/Desktop/Age_svm_c" + str(c) + "_linear.pdf")
    plt.show()
    print 
    print

results = pd.DataFrame(results, columns = ["C", "F1_Micro", "F1_Macro"])

plt.figure(figsize = (20,20))
sns.clustermap(cm_total, annot = True, cmap="YlGnBu", row_cluster=False, col_cluster=True, method="ward", annot_kws={"size": 8})
#plt.ylabel("True age group")
#plt.xlabel("Predicted age group")
plt.savefig("/Users/sklarjg/Desktop/Age_svm_c" + str(c) + "_linear_clustered.pdf")
plt.show()

#%%

metadata_df = metadata_df.loc[metadata_df.index.isin(age_cohort.index), :]
metadata_df["age_years"] = metadata_df["age_years"].astype(float)
metadata_df = metadata_df[(metadata_df["age_years"] >= 18.0) & (metadata_df["age_years"] <= 80.0) ]
otu_df = otu_df.loc[otu_df.index.isin(metadata_df.index), :]

metadata_df["age_years"].hist(bins = 30)
plt.show()

metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].map({"I tend to have normal formed stool - Type 3 and 4": "Normal", 
                                                                                   "I tend to have normal formed stool": "Normal",
                                                                                   "I tend to have diarrhea (watery stool) - Type 5, 6 and 7": "Loose",
                                                                                   "I tend to be constipated (have difficulty passing stool) - Type 1 and 2": "Hard",
                                                                                   "I tend to be constipated (have difficulty passing stool)": "Hard",
                                                                                   "I tend to have diarrhea (watery stool)": "Loose"})

#%%    
    
multiclasses = ["bowel_movement_frequency",
                "bowel_movement_quality",
                "census_region",
                "collection_season",
                "diet_type",
                "race",
                "sleep_duration",
                "types_of_plants",
                "fed_as_infant",
                "alcohol_frequency"]
results = []
for multic in multiclasses:
    curr_var_cohort = metadata_df[~metadata_df[multic].isin(["Not provided", "Unspecified", "Not sure",
                                                             5, "Other", "I don't know, I do not have a point of reference"])]
    curr_var_cohort.dropna(subset = [multic], inplace = True)
    class_counts = curr_var_cohort[multic].value_counts()
    max_class_size = np.median(class_counts.values)
    otu_cohort , meta_cohort = BalanceClasses(otu_df, curr_var_cohort, multic, int(max_class_size), 10)
    
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
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 10, random_state = 12345)

    alg_rf = SVC(C = 0.1, kernel='linear', class_weight = 'balanced')  
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



 #%%



labels = ["Never", "Rarely", "Occasionally", "Regularly", "Daily"]
cm_total.index = labels
cm_total.columns = labels
plt.figure(figsize = (15,12))
sns.heatmap(cm_total, annot = True, cmap="YlGnBu")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/multiclass/" + multic + "_linear.pdf")
plt.show()





