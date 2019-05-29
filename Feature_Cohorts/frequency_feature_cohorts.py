#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:55:00 2019

@author: sklarjg
"""
#import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)


removal_features = ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]
for val in removal_features:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
print len(metadata_df)

metadata_df.loc[:, ["bmi", "age_years"]] = metadata_df.loc[:, ["bmi", "age_years"]].astype(float)
metadata_df = metadata_df[metadata_df["country"].isin(['USA', 'United Kingdom', 'Canada'])]
metadata_df["bmi"] = metadata_df["bmi"].astype(float)
metadata_df["age_years"] = metadata_df["age_years"].astype(float)
metadata_df["weight_kg"] = metadata_df["weight_kg"].astype(float)
metadata_df["longitude"] = metadata_df["longitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)
metadata_df["latitude"] = metadata_df["latitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)
metadata_df["race"] = metadata_df["race"].replace(["Unspecified", "Not provided"], "Other")
metadata_df["diet_type"] = metadata_df["diet_type"].replace(["Unspecified", "Not provided"], "Other")
metadata_df['sex'] = metadata_df['sex'].replace(["Unspecified", "Not provided", "unspecified"], "other")


###MATCHING METADATA FEATURES TO BUILD BALANCED COHORTS
le_race = LabelEncoder()
le_sex = LabelEncoder()
cohort_matching_features = ["sex", "age_years", "bmi", "longitude", "latitude", "race", "diet_type", "alcohol_frequency"]
metadata_matching = metadata_df.loc[:, cohort_matching_features]
le_race.fit(metadata_matching["race"].unique())
metadata_matching["race"] = le_race.transform(metadata_matching["race"])
le_sex.fit(metadata_matching["sex"].unique())
metadata_matching["sex"] = le_sex.transform(metadata_matching["sex"])
metadata_matching["diet_type"] = metadata_matching["diet_type"].map({"Omnivore": 0,"Omnivore but do not eat red meat": 1,"Vegetarian but eat seafood": 2,
                                                                     "Vegetarian": 3,"Vegan": 4,"Other": 5})
scaler = StandardScaler()
metadata_matching= pd.DataFrame(scaler.fit_transform(metadata_matching.loc[:, cohort_matching_features].astype(float)), index = metadata_matching.index, columns = metadata_matching.columns)

freq_values = ['artificial_sweeteners','exercise_frequency','fermented_plant_frequency','flossing_frequency','frozen_dessert_frequency',
               'fruit_frequency', 'high_fat_red_meat_frequency','homecooked_meals_frequency','meat_eggs_frequency','milk_cheese_frequency',
               'milk_substitute_frequency','olive_oil','one_liter_of_water_a_day_frequency','poultry_frequency','prepared_meals_frequency',
               'probiotic_frequency','ready_to_eat_meals_frequency','red_meat_frequency', 'salted_snacks_frequency', 'seafood_frequency',
               'smoking_frequency', 'sugar_sweetened_drink_frequency', 'sugary_sweets_frequency','teethbrushing_frequency',
               'vegetable_frequency','vitamin_b_supplement_frequency','vitamin_d_supplement_frequency','whole_eggs',
               'whole_grain_frequency','alcohol_frequency']

def buildDataSubset(metadata, target_var, pos_target, neg_target, base_match):
    ###metadata: dataset holding all the target variables
    ###target_var: which frequency variable to make a balanced cohort for
    ###pos_target: daily-rare 
    ###neg_target: never consumption
    ###base_match: boolean (use the base matching variables (standard), or informed matching variables (base + diet and alcohol))
    ###
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
    if base_match:
        base_matching_features = ["sex", "age_years", "bmi", "longitude", "latitude", "race"]
        cm = cosine_similarity(metadata_matching.loc[pos_class, base_matching_features], metadata_matching.loc[neg_class, base_matching_features])
    else:
        cm = cosine_similarity(metadata_matching.loc[pos_class, :], metadata_matching.loc[neg_class, :])
    cm = pd.DataFrame(cm, index = pos_class, columns = neg_class)
    cohort = []
    for pos_index in cm.index:
        neg_match = cm.loc[pos_index,:].idxmax(axis = 1)
        cm.drop(neg_match, axis = 1, inplace = True)
        cohort.append(pos_index)
        cohort.append(neg_match)     
    return cohort

#%%
    
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 18.0) & (metadata_df["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["bmi"] < 30.0) & (metadata_df_healthy["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
base_matching_features = ["sex", "age_years", "bmi", "longitude", "latitude", "race"]
file_names = ["_rare_cohort.csv",
              "_occasional_cohort.csv",
              "_regular_cohort.csv",
              "_daily_cohort.csv"]
'''
#Standard Matching Cohorts
for var in freq_values:
    print var
    print metadata_df_healthy[var].value_counts()
    for group in range(1,5):
        print ("Constructing " + file_names[group -1].split("_")[1] +  "cohort: " + str(group))
        cohort = buildDataSubset(metadata_df_healthy, var, group, 0, True)
        cohort = metadata_matching.loc[cohort, :]
        cohort["target"] = metadata_df_healthy.loc[cohort.index, var]
        cohort.loc[cohort["target"] == group, "target"] = 1
        cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/frequency_cohorts/" + var + file_names[group -1])
    print
    print 
    print
 '''   
  
##Updated maching cohorts
for var in freq_values:
    print var
    print metadata_df_healthy[var].value_counts()
    for group in range(1,5):
        print ("Constructing " + file_names[group -1].split("_")[1] +  "cohort: " + str(group))
        cohort = buildDataSubset(metadata_df_healthy, var, group, 0, False)
        cohort = metadata_matching.loc[cohort, :]
        cohort["target"] = metadata_df_healthy.loc[cohort.index, var]
        cohort.loc[cohort["target"] == group, "target"] = 1
        cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/frequency_cohorts_updated/" + var + file_names[group -1])
    print
    print 
    print
    
    
    
    
    
    
    
    
    
    
    
    



























