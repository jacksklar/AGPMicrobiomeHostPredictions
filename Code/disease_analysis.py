#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:02:13 2019

@author: sklarjg
"""
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt

feature_groups = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_groups.csv", index_col = 0)
metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
feature_groups = feature_groups[~feature_groups.index.isin(['dna_extracted','physical_specimen_remaining','public', 'breastmilk_formula_ensure', 'acne_medication', 'acne_medication_otc', 'alcohol_consumption'])]
binary_features = feature_groups[feature_groups["group"].isin(["binary","disease"])].index.values

### Removal of samples that did not have valid annotation of important information used for matching
for val in ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
metadata_df.loc[:, ["bmi", "age_years"]] = metadata_df.loc[:, ["bmi", "age_years"]].astype(float)
metadata_df = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                          (metadata_df["age_years"] <= 80.0) &
                          (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                          (metadata_df["bmi"] >= 12.5) & 
                          (metadata_df["bmi"] <= 50.0) &
                          (metadata_df["alcohol_frequency"] != 5) &
                          (metadata_df["milk_cheese_frequency"] !=  5) &
                          (metadata_df["meat_eggs_frequency"] != 5) &
                          (metadata_df["vegetable_frequency"] != 5) &
                          (~metadata_df["bowel_movement_quality"].isin(["Unspecified", "Not provided", "I don't know, I do not have a point of reference"])) &
                          (metadata_df["country"].isin(["USA", "United Kingdom", "Canada"]))]

#%%

disease_stat = metadata_df.loc[:, ["acid_reflux", "add_adhd", "alzheimers", "asd", "autoimmune", "cancer", "cardiovascular_disease", "cdiff", "clinical_condition", 
                                   "depression_bipolar_schizophrenia", "diabetes", "epilepsy_or_seizure_disorder", "fungal_overgrowth", "ibd", "ibs", "kidney_disease",
                                   "liver_disease", "lung_disease", "migraine", "pku", "sibo", "skin_condition", "thyroid"]]
for disease in disease_stat.columns:
    disease_stat[disease] = metadata_df[disease].replace([2, 3, 4], 0)    
disease_coo = disease_stat.astype(int)
disease_coo = disease_coo.T.dot(disease_coo)
disease_coo.to_csv("/Users/sklarjg/Desktop/disease_co_occurrance.csv")
np.fill_diagonal(disease_coo.values, 0)
sns.clustermap(disease_coo)
plt.savefig("/Users/sklarjg/Desktop/disease_co_occurrance_mat.pdf")
disease_sum = disease_stat.sum(axis = 0)
disease_sum.to_csv("/Users/sklarjg/Desktop/disease_sum.csv")

#%%

df = pd.read_csv("/Users/sklarjg/Desktop/disease_co_occurrance_2.csv", index_col = 0)
sns.heatmap(df)
plt.savefig("/Users/sklarjg/Desktop/disease_co_occurrance_mat_2.pdf")
sns.clustermap(df)
plt.savefig("/Users/sklarjg/Desktop/disease_co_occurrance_mat_3.pdf")