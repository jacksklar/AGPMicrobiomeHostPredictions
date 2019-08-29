#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:14:31 2019

@author: sklarjg
"""

import numpy as np 
import pandas as pd 
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


def LogMeanFoldChange(feature_list, cohort_path, save_path, file_name):
    all_features_foldchange = pd.DataFrame([], columns = otu_df.columns.values)
    for feature in feature_list:
        if feature.split(".")[0] == "":
            continue
        feature_name = feature.split(".")[0]
        cohort = pd.read_csv(cohort_path + feature, index_col = 0)
        if cohort.columns.contains("target"):
            print feature_name
            pos_class = cohort[cohort["target"] == 1].index
            neg_class = cohort[cohort["target"] == 0].index
            feature_fold_change = pd.Series([])
            for otu in otu_df.columns.values:
                mean_pos = otu_df.loc[pos_class, otu].mean()
                mean_neg = otu_df.loc[neg_class, otu].mean()
                mean_fold_change = np.log(mean_pos + 1.0) - np.log(mean_neg + 1.0)
                feature_fold_change[otu] = mean_fold_change
            all_features_foldchange.loc[feature_name,:] = feature_fold_change
    all_features_foldchange.to_csv(save_path + file_name)


#%%

cohort_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/binary_cohorts_no_matching/"
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_no_matching/"
feature_list = os.listdir(cohort_path)
file_name = "binary_cohorts_no_matching_lmfc.csv"
feature_remove = feature_info[feature_info["type"].isin(["Race","other"])].index.values
feature_remove = [feature + ".csv" for feature in feature_remove]
feature_list = [feature for feature in feature_list if feature not in feature_remove]
LogMeanFoldChange(feature_list, cohort_path, save_path, file_name)

cohort_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/binary_cohorts_standard/"
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_standard/"
feature_list = os.listdir(cohort_path)
file_name = "binary_cohorts_standard_lmfc.csv"
feature_remove = feature_info[feature_info["type"].isin(["Race","other"])].index.values
feature_remove = [feature + ".csv" for feature in feature_remove]
feature_list = [feature for feature in feature_list if feature not in feature_remove]
LogMeanFoldChange(feature_list, cohort_path, save_path, file_name)



#%%

cohort_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_cohorts/"
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/"
frequency_list = os.listdir(cohort_path)
filename = "frequency_cohorts_lmfc.csv"

LogMeanFoldChange(frequency_list, cohort_path, save_path, file_name)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
