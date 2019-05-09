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
print len(metadata_df)
matching_features = ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]
for val in matching_features:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
print len(metadata_df)
metadata_df.loc[:, ["bmi", "age_years"]] = metadata_df.loc[:, ["bmi", "age_years"]].astype(float)

###keep samples from countries with sufficient number of samples 
metadata_df = metadata_df[metadata_df["country"].isin(['USA', 'United Kingdom', 'Australia', 'Canada'])]
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

cohort_matching_features = ["sex", "age_years", "bmi", "longitude", "latitude", "race"]
metadata_matching = metadata_df.loc[:, cohort_matching_features]

le_race.fit(metadata_matching["race"].unique())
metadata_matching["race"] = le_race.transform(metadata_matching["race"])

le_sex.fit(metadata_matching["sex"].unique())
metadata_matching["sex"] = le_sex.transform(metadata_matching["sex"])

metadata_matching = metadata_matching.loc[:, cohort_matching_features]
scaler = StandardScaler()
metadata_matching= pd.DataFrame(scaler.fit_transform(metadata_matching.loc[:, cohort_matching_features].astype(float)), index = metadata_matching.index, columns = metadata_matching.columns)

def buildDataSubset(metadata, target_var, pos_target, neg_target):
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
    cm = cosine_similarity(metadata_matching.loc[pos_class, :], metadata_matching.loc[neg_class, :])
    cm = pd.DataFrame(cm, index = pos_class, columns = neg_class)
    #cohort_pairs = []
    cohort = []
    for pos_index in cm.index:
        neg_match = cm.loc[pos_index,:].idxmax(axis = 1)
        cm.drop(neg_match, axis = 1, inplace = True)
        cohort.append(pos_index)
        cohort.append(neg_match)     
    return cohort

    

freq_values = ['artificial_sweeteners',
               'exercise_frequency',
               'fermented_plant_frequency',
               'flossing_frequency',
               'frozen_dessert_frequency',
               'fruit_frequency',
               'high_fat_red_meat_frequency',
               'homecooked_meals_frequency',
               'meat_eggs_frequency',
               'milk_cheese_frequency',
               'milk_substitute_frequency',
               'olive_oil',
               'one_liter_of_water_a_day_frequency',
               'poultry_frequency',
               'prepared_meals_frequency',
               'probiotic_frequency',
               'ready_to_eat_meals_frequency',
               'red_meat_frequency',
               'salted_snacks_frequency',
               'seafood_frequency',
               'smoking_frequency',
               'sugar_sweetened_drink_frequency',
               'sugary_sweets_frequency',
               'teethbrushing_frequency',
               'vegetable_frequency',
               'vitamin_b_supplement_frequency',
               'vitamin_d_supplement_frequency',
               'whole_eggs',
               'whole_grain_frequency',
               'alcohol_frequency']

'''
'never':  0
'rarely (a few times/month)':  1
'occasionally (1-2 times/week)': 2
'regularly (3-5 times/week)': 3
'daily': 4
"unspecified": 5
'''

#%%
    
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 18.0) & (metadata_df["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["bmi"] < 30.0) & (metadata_df_healthy["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]

#%%

file_names = ["_rare_cohort.csv",
              "_occasional_cohort.csv",
              "_regular_cohort.csv",
              "_daily_cohort.csv"]

for var in freq_values:
    print var
    print metadata_df_healthy[var].value_counts()
    for group in range(1,5):
        print ("Constructing " + file_names[group -1].split("_")[1] +  "cohort: " + str(group))
        cohort = buildDataSubset(metadata_df_healthy, var, group, 0)
        cohort = metadata_matching.loc[cohort, :]
        cohort["target"] = metadata_df_healthy.loc[cohort.index, var]
        cohort.loc[cohort["target"] == group, "target"] = 1
        cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/Frequency_cohorts/" + var + file_names[group -1])
    print
    print 
    print




























