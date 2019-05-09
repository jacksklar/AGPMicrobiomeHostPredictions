#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:51:58 2019

@author: sklarjg
"""
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

feature_groups = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_groups.csv", index_col = 0)
metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
feature_groups = feature_groups[~feature_groups.index.isin(['dna_extracted','physical_specimen_remaining','public', 
                                                            'assigned_from_geo','livingwith','roommates_in_study', 
                                                            'breastmilk_formula_ensure', 'acne_medication', 'acne_medication_otc', 
                                                            'alcohol_consumption'])]
binary_features = feature_groups[feature_groups["group"].isin(["binary","disease"])].index.values
other_features = feature_groups[feature_groups["group"] == 'other'].index.values

###Selecting Healthy Samples:
matching_features = ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]
for val in matching_features:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
metadata_df.loc[:, ["bmi", "age_years"]] = metadata_df.loc[:, ["bmi", "age_years"]].astype(float)

metadata_df = metadata_df[(metadata_df["age_years"] > 17.0) & (metadata_df["age_years"] <= 80.0)]
metadata_df = metadata_df[(metadata_df["bmi"] < 30.0) & (metadata_df["bmi"] > 12.5)]
metadata_df = metadata_df[metadata_df["ibd"] == 0]
metadata_df = metadata_df[metadata_df["diabetes"] == 0]
metadata_df = metadata_df[metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
metadata_df = metadata_df[metadata_df["country"].isin(["USA", "United Kingdom", "Australia", "Canada"])]

###MATCHING METADATA FEATURES TO BUILD BALANCED COHORTS
le_race = LabelEncoder()
le_diet = LabelEncoder()
le_sex = LabelEncoder()
mid_america_log = -100
mid_america_lat = 40

metadata_df["age_years"] = metadata_df["age_years"].astype(float)
metadata_df["bmi"] = metadata_df["bmi"].astype(float)
metadata_df["weight_kg"] = metadata_df["weight_kg"].astype(float)
metadata_df["longitude"] = metadata_df["longitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)
metadata_df["latitude"] = metadata_df["latitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)

metadata_df["race"] = metadata_df["race"].replace(["Unspecified", "Not provided"], "Other")
le_race.fit(metadata_df["race"].unique())
metadata_df["race"] = le_race.transform(metadata_df["race"])

metadata_df['sex'] = metadata_df['sex'].replace(["Unspecified", "Not provided", "unspecified"], "other")
le_sex.fit(metadata_df["sex"].unique())
metadata_df["sex"] = le_sex.transform(metadata_df["sex"])

cohort_matching_features = ["sex", "age_years", "bmi", "longitude", "latitude", "race"]
metadata_matching = metadata_df.loc[:, cohort_matching_features]

scaler = StandardScaler()
metadata_matching= pd.DataFrame(scaler.fit_transform(metadata_df.loc[:, cohort_matching_features].astype(float)), index = metadata_df.index, columns = cohort_matching_features)

#%%


def buildDataSubset(target_name):
    target = metadata_df[target_name].astype(int)  
    pos_class = target[target == 1].index
    neg_class = target[target == 0].index
    n_pos = len(pos_class)
    n_neg = len(neg_class)
    if n_pos > n_neg: 
        temp = pos_class
        pos_class = neg_class
        neg_class = temp
    cm = cosine_similarity(metadata_matching.loc[pos_class, :], metadata_matching.loc[neg_class, :])
    cm = pd.DataFrame(cm, index = pos_class, columns = neg_class)
    cohort_pairs = []
    cohort = []
    for pos_index in cm.index:
        neg_match = cm.loc[pos_index,:].idxmax(axis = 1)
        cm.drop(neg_match, axis = 1, inplace = True)
        cohort.append(pos_index)
        cohort.append(neg_match)
        cohort_pairs.append([pos_index, neg_match])
    cohort = metadata_matching.loc[cohort, :]
    cohort.loc[cohort.index, "target"] = metadata_df.loc[cohort.index, target_name]
    cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Feature_cohorts/" + str(target_name) + ".csv")

for feature in binary_features:
    feature_counts = metadata_df[feature].value_counts()
    if 1 not in feature_counts.index:
        continue
    n_feature_positive = feature_counts[1]
    if n_feature_positive < 50:
        print "       Not enough samples", feature, str(n_feature_positive)
        continue
    else:
        print feature, str(n_feature_positive)
        buildDataSubset(feature)
            
            
            
            
            




