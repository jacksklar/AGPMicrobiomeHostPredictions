#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:14:31 2019

@author: sklarjg
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns        
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

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
otu_df = otu_df.loc[metadata_df.index, :]
taxa_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/taxa_md5.xls", sep = "\t", index_col = 0)
taxa_df = taxa_df[taxa_df.index.isin(otu_df.columns)]
taxa_df = taxa_df.replace(np.nan, 'Unknown', regex=True)
otu_df = mapOTU(otu_df, taxa_df, "Genus")
otu_df = otu_df.reindex(otu_df.mean().sort_values(ascending = False).index, axis=1)
otu_df = np.log(otu_df + 1)

#%%

dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Feature_cohorts/"
imp_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Genus_BoostedT/Importances/"
fold_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Genus_BoostedT/Fold_change/"
feature_list = os.listdir(dir_path)

for feature in feature_list:
    if feature.split(".")[0] == "":
        continue
    exists = os.path.isfile(imp_path + feature)
    if exists:
        print feature
        cohort = pd.read_csv(dir_path + feature, index_col = 0)
        
        pos_class = cohort[cohort["target"] == 1].index
        neg_class = cohort[cohort["target"] == 0].index
        importances = pd.read_csv(imp_path + feature, header = None, index_col = 0)[1]        
        feature_fold_change = pd.Series([])
        for otu in importances.index.values:
            mean_pos = otu_df.loc[pos_class, otu].mean()
            mean_neg = otu_df.loc[neg_class, otu].mean()
            mean_fold_change = mean_pos - mean_neg
            feature_fold_change[otu] = mean_fold_change
        feature_fold_change.to_csv(fold_path + feature)
    
#%%
dir_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Frequency_cohorts/"
imp_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Genus_Frequency/Importances/"
fold_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Genus_Frequency/Fold_change/"
frequency_list = os.listdir(dir_path)

frequency_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/frequency_feature_info.csv", index_col = 0)

for feature in frequency_info["Variable"].unique():
    cohort_filenames = frequency_info[frequency_info["Variable"] == feature].index.values + ".csv"
    freq_names = [ val.split("_")[-2] for val in cohort_filenames]
    print feature
    print 
    print    
    feature_fold_change = pd.DataFrame([], columns = ['daily', 'regular', 'occasional', 'rare']) 
    
    for filename, freq_name in zip(cohort_filenames, freq_names):
        print filename, freq_name
        exists = os.path.isfile(imp_path + filename)
        if exists:
            cohort = pd.read_csv(dir_path + filename, index_col = 0)
            pos_class = cohort[cohort["target"] == 1].index
            neg_class = cohort[cohort["target"] == 0].index
            importances = pd.read_csv(imp_path + filename, header = None, index_col = 0)[1]        
            for otu in importances.index.values:
                mean_pos = otu_df.loc[pos_class, otu].mean()
                mean_neg = otu_df.loc[neg_class, otu].mean()
                mean_fold_change = mean_pos - mean_neg
                feature_fold_change.loc[otu, freq_name] = mean_fold_change
    feature_fold_change.to_csv(fold_path + feature + ".csv") 
            


#%%

for feature in frequency_list:
    exists = os.path.isfile(imp_path + feature)
    if exists:
        print feature
        cohort = pd.read_csv(dir_path + feature, index_col = 0)
        
        pos_class = cohort[cohort["target"] == 1].index
        neg_class = cohort[cohort["target"] == 0].index
        importances = pd.read_csv(imp_path + feature, header = None, index_col = 0)[1]        
        feature_fold_change = pd.Series([])
        for otu in importances.index.values:
            mean_pos = otu_df.loc[pos_class, otu].mean()
            mean_neg = otu_df.loc[neg_class, otu].mean()
            mean_fold_change = mean_pos - mean_neg
            feature_fold_change[otu] = mean_fold_change
        feature_fold_change.to_csv(fold_path + feature)