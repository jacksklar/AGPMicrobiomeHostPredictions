#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:42:27 2019

@author: sklarjg
"""

#import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dir_path = "/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/"

metadata_df = pd.read_csv(dir_path + "Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
print("Full Population size: ", str(len(metadata_df)))
for val in ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
print("Population size (missing metadata removed): ", str(len(metadata_df)))

metadata_df = metadata_df[metadata_df["country"].isin(['USA', 'United Kingdom', 'Canada'])]
metadata_df["longitude"] = metadata_df["longitude"].replace(["Unspecified", "Not provided"], -10.0)#.astype(float)
metadata_df["latitude"] = metadata_df["latitude"].replace(["Unspecified", "Not provided"], -10.0)#.astype(float)
metadata_df["race"] = metadata_df["race"].replace(["Unspecified", "Not provided"], "Other")
metadata_df['sex'] = metadata_df['sex'].replace(["Unspecified", "Not provided", "unspecified"], "other")
metadata_df['diet_type'] = metadata_df['diet_type'].replace(["Unspecified", "Not provided"], "Other")
metadata_df.loc[:, ["bmi", "age_years", "weight_kg", "longitude", "latitude"]] = metadata_df.loc[:, ["bmi", "age_years", "weight_kg", "longitude", "latitude"]].astype(float)


#Samples with missing geographic locations get the centroid of their country or residence
usa_missing_geo = metadata_df[(metadata_df["longitude"] == -10.0) & (metadata_df["country"] == "USA")].index.values
uk_missing_geo = metadata_df[(metadata_df["longitude"] == -10.0) & (metadata_df["country"] == "United Kingdom")].index.values
can_missing_geo = metadata_df[(metadata_df["longitude"] == -10.0) & (metadata_df["country"] == "Canada")].index.values
metadata_df.loc[usa_missing_geo, ["longitude", "latitude"]] = (-98.6, 39.8)
metadata_df.loc[uk_missing_geo, ["longitude", "latitude"]] = (-1.5, 52.6)
metadata_df.loc[can_missing_geo, ["longitude", "latitude"]] = (-79.4, 43.9)
metadata_df = metadata_df.loc[metadata_df["bmi"] <= 60, :]

###MATCHING METADATA FEATURES TO BUILD BALANCED COHORTS
le_race = LabelEncoder()
le_sex = LabelEncoder()


cohort_matching_features = ["longitude", "latitude"]
cohort_feats = ["sex", "age_years", "bmi", "bmi_cat", "country", "longitude", "latitude", "race", "ibd", "diabetes", "antibiotic_history"]

metadata_matching = metadata_df.loc[:, cohort_feats]
le_race.fit(metadata_matching["race"].unique())
metadata_matching["race"] = le_race.transform(metadata_matching["race"])
le_sex.fit(metadata_matching["sex"].unique())
metadata_matching["sex"] = le_sex.transform(metadata_matching["sex"])
metadata_matching["antibiotic_history"] = metadata_matching["antibiotic_history"].map({"Year": 0, "I have not taken antibiotics in the past year.": 1, "6 months": 2, "Month": 3, "Week": 4})
metadata_matching = metadata_matching.loc[:, cohort_matching_features]
scaler = StandardScaler()
metadata_matching= pd.DataFrame(scaler.fit_transform(metadata_matching.loc[:, cohort_matching_features].astype(float)), index = metadata_matching.index, columns = metadata_matching.columns)


def buildDataSubset(metadata, target_var, pos_target, neg_target, age_related, bmi_related, location_related):
    ##target_var: name of metadata variable
    ##pos_target: positive class label
    ##neg_target: negavtive class label (possibly a list of labels)
    target = metadata[target_var]
    if type(pos_target) == list:
        pos_class = target[target.isin(pos_target)].index
    else:
        pos_class = target[target == pos_target].index
    if type(neg_target) == list:
        neg_class = target[target.isin(neg_target)].index
    else:
        neg_class = target[target == neg_target].index
    n_pos = len(pos_class)
    n_neg = len(neg_class)
    if n_pos > n_neg: 
        temp = pos_class
        pos_class = neg_class
        neg_class = temp
    cm = euclidean_distances(metadata_matching.loc[pos_class, :], metadata_matching.loc[neg_class, :])
    cm = pd.DataFrame(cm, index = pos_class, columns = neg_class)
    cohort = []        
    distances = []
    worst_distances = []
    for pos_index in cm.index:
        neg_match = cm.loc[pos_index,:].idxmin(axis = 1)
        worst_match = cm.loc[pos_index,:].idxmax(axis = 1)
        dist = cm.loc[pos_index, neg_match]
        worst_dist = cm.loc[pos_index, worst_match]
        cm.drop(neg_match, axis = 1, inplace = True)
        cohort.append(pos_index)
        cohort.append(neg_match)
        distances.append(dist)
        distances.append(dist)
        worst_distances.append(worst_dist)
        worst_distances.append(worst_dist)
    cohort = metadata.loc[cohort, cohort_feats]
    cohort["target"] = metadata.loc[cohort.index, target_var]
    if pos_target != 1:
        cohort["target"] = cohort["target"].map({pos_target: 1, neg_target: 0})
    cohort["pairID"] = pairIDs(len(cohort))
    cohort["pairDist"] = distances
    cohort["worstPairDist"] = worst_distances
    return cohort


def pairIDs(length):
    num_pairs = int(length / 2)
    pair_ids = []
    for val in range(num_pairs):
        pair_ids.append(val)
        pair_ids.append(val)
    return pair_ids    

#%%

agp_healthy_population = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                     (metadata_df["age_years"] <= 69.0) &
                                     (metadata_df["antibiotic_history"] == "I have not taken antibiotics in the past year.") &
                                     (metadata_df["ibd"] == 0) &
                                     (metadata_df["diabetes"] == 0) &
                                     (metadata_df["bmi"] >= 18.5) & 
                                     (metadata_df["bmi"] < 30.0)]
print("AGP Healthy Population size: ", len(agp_healthy_population))



#%%

###BMI SPECIAL FEATURE:
healthy_population = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                     (metadata_df["age_years"] <= 69.0) &
                                     (metadata_df["antibiotic_history"] == "I have not taken antibiotics in the past year.") &
                                     (metadata_df["ibd"] == 0) &
                                     (metadata_df["diabetes"] == 0) &
                                     (metadata_df["bmi"] <= 60.0) &
                                     (metadata_df["bmi"] >= 12.5)]
print(healthy_population["bmi_cat"].value_counts())


print("Constructing Obese cohort with matched normal samples")
obese_cohort = buildDataSubset(healthy_population, "bmi_cat", "Obese", "Normal", False, True, False)
obese_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/Obese_cohort.csv")

print("Constructing Overweight cohort with matched normal samples")
overweight_cohort = buildDataSubset(healthy_population, "bmi_cat", "Overweight", "Normal", False, True, False)
overweight_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/Overweight_cohort.csv")

print("Constructing Underweight cohort with matched normal samples")
underweight_cohort = buildDataSubset(healthy_population, "bmi_cat", "Underweight", "Normal", False, True, False)
underweight_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/Underweight_cohort.csv")


###IBD
healthy_population = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                     (metadata_df["age_years"] <= 69.0) &
                                     (metadata_df["antibiotic_history"] == "I have not taken antibiotics in the past year.") &
                                     (metadata_df["diabetes"] == 0) &
                                     (metadata_df["bmi"] >= 18.5) & 
                                     (metadata_df["bmi"] < 30.0)]

print("Constructing IBD cohort with matched healthy samples")
ibd_cohort = buildDataSubset(healthy_population, "ibd", 1, 0, False, False, False)
ibd_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/IBD_cohort.csv")

##Antibiotic History
healthy_population = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                     (metadata_df["age_years"] <= 69.0) &
                                     (metadata_df["ibd"] == 0) &
                                     (metadata_df["diabetes"] == 0) &
                                     (metadata_df["bmi"] >= 18.5) & 
                                     (metadata_df["bmi"] < 30.0)]
print(healthy_population["antibiotic_history"].value_counts())

print("Constructing Antiobiotic Year cohort with matched healthy samples")
antiB_cohort1 = buildDataSubset(healthy_population, "antibiotic_history", "Year", "I have not taken antibiotics in the past year.", False, False, False)
antiB_cohort1.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/antiB_Year_cohort.csv")

print("Constructing Antiobiotic 6 Months cohort with matched healthy samples")
antiB_cohort2 = buildDataSubset(healthy_population, "antibiotic_history", "6 months", "I have not taken antibiotics in the past year.", False, False, False)
antiB_cohort2.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/antiB_6Month_cohort.csv")

print("Constructing Antiobiotic Year cohort with matched healthy samples")
antiB_cohort3 = buildDataSubset(healthy_population, "antibiotic_history", "Month", "I have not taken antibiotics in the past year.", False, False, False)
antiB_cohort3.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/antiB_Month_cohort.csv")

print("Constructing Antiobiotic Week cohort with matched healthy samples")
antiB_cohort4 = buildDataSubset(healthy_population, "antibiotic_history", "Week", "I have not taken antibiotics in the past year.", False, False, False)
antiB_cohort4.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/antiB_Week_cohort.csv")

#%%

##Diabetes
healthy_population = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                     (metadata_df["age_years"] <= 69.0) &
                                     (metadata_df["antibiotic_history"] == "I have not taken antibiotics in the past year.") &
                                     (metadata_df["ibd"] == 0) &
                                     (metadata_df["bmi"] >= 18.5) & 
                                     (metadata_df["bmi"] <= 60.0)]

print(healthy_population["diabetes_type"].value_counts())

print("Constructing Diabetes type II cohort with matched healthy samples")
healthy_population.loc[healthy_population["diabetes_type"].isin(["Not provided", "Unspecified"]), "diabetes_type"] = 0
diabetes_cohort = buildDataSubset(healthy_population, "diabetes_type", "Type II diabetes", 0, False, False, False)
diabetes_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/diabetes_typeII_cohort.csv")

##Age
healthy_population = metadata_df[(metadata_df["antibiotic_history"] == "I have not taken antibiotics in the past year.") &
                                 (metadata_df["ibd"] == 0) &
                                 (metadata_df["diabetes"] == 0) &
                                 (metadata_df["bmi"] >= 18.5) & 
                                 (metadata_df["bmi"] < 30.0)]


print("Constructing young cohort with matched healthy adults")
healthy_population_young = healthy_population[healthy_population["age_years"] <= 69.0]
healthy_population_young["age_group"] = healthy_population_young["age_years"] 
young = healthy_population_young[healthy_population_young["age_years"] <= 19.0].index
old = healthy_population_young[healthy_population_young["age_years"] >= 20.0].index
healthy_population_young.loc[young, "age_group"] = 1
healthy_population_young.loc[old, "age_group"] = 0

young_cohort = buildDataSubset(healthy_population_young, "age_group", 1, 0, True, True, False)
young_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/age_young_cohort.csv")

print("Constructing 70 and over cohort with matched healthy adults")
healthy_population_old = healthy_population[healthy_population["age_years"] >= 20.0]
healthy_population_old["age_group"] = healthy_population_old["age_years"] 
young = healthy_population_old[healthy_population_old["age_years"] <= 69.0].index
old = healthy_population_old[healthy_population_old["age_years"] >= 70.0].index
healthy_population_old.loc[young, "age_group"] = 0
healthy_population_old.loc[old, "age_group"] = 1

old_cohort = buildDataSubset(healthy_population_old, "age_group", 1, 0, True, False, False)
old_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/age_old_cohort.csv")

healthy_population = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                     (metadata_df["age_years"] <= 69.0) &
                                     (metadata_df["antibiotic_history"] == "I have not taken antibiotics in the past year.") &
                                     (metadata_df["diabetes"] == 0) &
                                     (metadata_df["ibd"] == 0) &
                                     (metadata_df["bmi"] >= 18.5) & 
                                     (metadata_df["bmi"] < 30.0)]

print("Constructing Country cohort with matched healthy samples")
country_cohort = buildDataSubset(healthy_population, "country", "USA", "United Kingdom", False, False, False)
country_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/country_cohort.csv")
print(len(country_cohort))

