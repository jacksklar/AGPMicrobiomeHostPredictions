#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:37:13 2019

@author: sklarjg
"""

import os
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns        


metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
file_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/AGP_File_Metadata.csv", index_col = 0)
otufull_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/AGP_merged2_s50_rar10k_forR_md5.txt", sep = "\t").T

print "Raw Fastq reads downloaded from ENA:", len(file_df)
print "Participants (after rarification):", len(otufull_df)
print "Participants (no duplicates, only gut samples):", len(metadata_df)

matching_features = ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]
for val in matching_features:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
metadata_df.loc[:, ["bmi", "age_years"]] = metadata_df.loc[:, ["bmi", "age_years"]].astype(float)
print "Metadata (removed samples with unspecified matching metadata [diabetes, age_years, bmi, ibd, antibiotic_history]):", len(metadata_df)

###keep samples from countries with sufficient number of samples 
metadata_df = metadata_df[metadata_df["country"].isin(['USA', 'United Kingdom', 'Australia', 'Canada'])]
print "Metadata (samples from Countries: USA, UK, AUS, CAN):", len(metadata_df)

metadata_df_healthy = metadata_df[(metadata_df["bmi"] < 30.0) & (metadata_df["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[~metadata_df_healthy["age_years"].isin(["Not provided", "Unspecified"])]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]


print "Metadata Healthy Control Poputlation:", len(metadata_df_healthy)


#%%
cohort_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/age_cohorts/"
cohort_list = os.listdir(cohort_path)
cohort_list.remove(".DS_Store")

list_df = pd.DataFrame([[cohort, int(cohort.split(".")[0].split("r")[1])] for cohort in cohort_list]).sort_values(1, ascending  = True)
cohort_order = list_df.iloc[:, 0]

for n, cohort_name in enumerate(cohort_order):
    print cohort_name
    cohort = pd.read_csv(cohort_path + cohort_name, index_col = 0)
    test = metadata_df.loc[cohort[cohort["target"] == 1].index, "age_years"].astype(float)
    control = metadata_df.loc[cohort[cohort["target"] == 0].index, "age_years"].astype(float)
    print test.value_counts()
    
