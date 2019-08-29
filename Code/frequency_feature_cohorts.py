#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:55:00 2019

@author: sklarjg
"""
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)




for val in ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
    
metadata_df.loc[:, ["bmi", "age_years"]] = metadata_df.loc[:, ["bmi", "age_years"]].astype(float)

## define set of healthy population for construction of binary questionnaire variables:
##      - adults (20:80), with antibiotic use in the last 6-months or later, no IBD or Diabetes, excluding obese participants
##      - only include most represented countries (US, UK, Can)
metadata_df = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                          (metadata_df["age_years"] <= 80.0) &
                          (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                          (metadata_df["ibd"] == 0) &
                          (metadata_df["diabetes"] == 0) &
                          (metadata_df["bmi"] >= 12.5) & 
                          (metadata_df["bmi"] <= 40.0) &
                          (metadata_df["alcohol_frequency"] != 5) &
                          (metadata_df["milk_cheese_frequency"] !=  5) &
                          (metadata_df["meat_eggs_frequency"] != 5) &
                          (metadata_df["vegetable_frequency"] != 5) &
                          (~metadata_df["bowel_movement_quality"].isin(["Unspecified", "Not provided", "I don't know, I do not have a point of reference"])) &
                          (metadata_df["country"].isin(["USA", "United Kingdom", "Canada"]))]    

### preprocess host metadata used for matching: encode as integers to calculate cosine similarity matrix
metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].replace(["I tend to have diarrhea (watery stool)", "I tend to have diarrhea (watery stool) - Type 5, 6 and 7"], 0)
metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].replace(["I tend to have normal formed stool", "I tend to have normal formed stool - Type 3 and 4"], 1)
metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].replace(["I tend to be constipated (have difficulty passing stool)", "I tend to be constipated (have difficulty passing stool) - Type 1 and 2"], 2)

metadata_df["race"] = metadata_df["race"].replace(["Unspecified", "Not provided"], "Other")
metadata_df['sex'] = metadata_df['sex'].replace(["Unspecified", "Not provided", "unspecified"], "Other")
metadata_df["diet_type"] = metadata_df["diet_type"].replace(["Unspecified", "Not provided"], "Other")
metadata_df["longitude"] = metadata_df["longitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)
metadata_df["latitude"] = metadata_df["latitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)

le_sex = LabelEncoder()
le_sex.fit(metadata_df["sex"].unique())
metadata_df["sex"] = le_sex.transform(metadata_df["sex"])

## Samples with missing geographic locations get the centroid of their country or residence
usa_missing_geo = metadata_df[(metadata_df["longitude"] == -10.0) & (metadata_df["country"] == "USA")].index.values
uk_missing_geo = metadata_df[(metadata_df["longitude"] == -10.0) & (metadata_df["country"] == "United Kingdom")].index.values
can_missing_geo = metadata_df[(metadata_df["longitude"] == -10.0) & (metadata_df["country"] == "Canada")].index.values
metadata_df.loc[usa_missing_geo, ["longitude", "latitude"]] = (-98.6, 39.8)
metadata_df.loc[uk_missing_geo, ["longitude", "latitude"]] = (-1.5, 52.6)
metadata_df.loc[can_missing_geo, ["longitude", "latitude"]] = (-79.4, 43.9)


base_matching_features = ["longitude", "latitude"]
new_matching_features = ["sex", "age_years", "bmi", "longitude", "latitude", "alcohol_frequency", "milk_cheese_frequency", "bowel_movement_quality", "meat_eggs_frequency",  "vegetable_frequency"] #"meat_eggs_frequency", 
cohort_feats = ["sex", "age_years", "bmi", "bmi_cat", "country", "longitude", "latitude", "race", "antibiotic_history", "diet_type",  "alcohol_frequency","milk_cheese_frequency", "bowel_movement_quality", "meat_eggs_frequency", "vegetable_frequency"]

##Standard Scale all feautres so they have equal impact on the cosine similarity calculations between positive class (cases), and the healthy control population
metadata_matching = metadata_df.loc[:, new_matching_features]
scaler = StandardScaler()
metadata_matching= pd.DataFrame(scaler.fit_transform(metadata_df.loc[:, new_matching_features].astype(float)), index = metadata_df.index, columns = new_matching_features)


## annotate Adjacent pairs in the cohort dataframe as matched paris
def pairIDs(length):
    num_pairs = length/2
    pair_ids = []
    for val in range(num_pairs):
        pair_ids.append(val)
        pair_ids.append(val)
    return pair_ids


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
        base_matching_features = ["longitude", "latitude"]
        cm = euclidean_distances(metadata_matching.loc[pos_class, base_matching_features], metadata_matching.loc[neg_class, base_matching_features])
    else:
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
        
    cohort = metadata_df.loc[cohort, cohort_feats]
    cohort["target"] = metadata_df.loc[cohort.index, target_var]
    cohort.loc[cohort["target"] == pos_target, "target"] = 1
    cohort["pairID"] = pairIDs(len(cohort))
    cohort["pairDist"] = distances
    cohort["worstPairDist"] = worst_distances
    return cohort

#%%
    
        
freq_values = ['artificial_sweeteners','exercise_frequency','fermented_plant_frequency','frozen_dessert_frequency',
               'fruit_frequency', 'high_fat_red_meat_frequency','homecooked_meals_frequency','meat_eggs_frequency','milk_cheese_frequency',
               'milk_substitute_frequency','olive_oil','one_liter_of_water_a_day_frequency','poultry_frequency','prepared_meals_frequency',
               'probiotic_frequency','ready_to_eat_meals_frequency','red_meat_frequency', 'salted_snacks_frequency', 'seafood_frequency',
               'smoking_frequency', 'sugar_sweetened_drink_frequency', 'sugary_sweets_frequency',
               'vegetable_frequency','vitamin_b_supplement_frequency','vitamin_d_supplement_frequency','whole_eggs',
               'whole_grain_frequency','alcohol_frequency']        
        
file_names = ["_rare_cohort.csv",
              "_occasional_cohort.csv",
              "_regular_cohort.csv",
              "_daily_cohort.csv"]


for var in freq_values:
    group_counts = metadata_df[var].value_counts()
    if group_counts[4] <= 200:
        print "      Daily group: ",  group_counts[4], group_counts[4] + group_counts[3]
        metadata_df.loc[metadata_df[var] == 4, var] = 3
        print metadata_df[var].value_counts()
        print
    if group_counts[0] <= 200:
        print "      Never group: ",  group_counts[0], group_counts[0] + group_counts[1]
        metadata_df.loc[metadata_df[var] == 1, var] = 0
        print metadata_df[var].value_counts()
        print 


#%%

##unmatched
for var in freq_values:
    print var
    frequency_groups = np.sort(metadata_df[var].unique())
    if 5 in frequency_groups:
        print "five"
        frequency_groups = np.delete(frequency_groups, np.argwhere(frequency_groups == 5))
    control_value = frequency_groups[0]
    case_groups = frequency_groups[1:]
    print control_value, case_groups
    print
    print 
    for group in case_groups:
        print ("Constructing " + file_names[group -1].split("_")[1] +  "cohort: " + str(group))
        cohort = buildDataSubset(metadata_df, var, group, control_value, True)
        cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/frequency_cohorts_no_matching/" + var + file_names[group -1])
    print
    print 
    print    
#%%

##Matched    
for var in freq_values:
    print var
    frequency_groups = np.sort(metadata_df[var].unique())
    if 5 in frequency_groups:
        print "five"
        frequency_groups = np.delete(frequency_groups, np.argwhere(frequency_groups == 5))
    control_value = frequency_groups[0]
    case_groups = frequency_groups[1:]
    print control_value, case_groups
    print
    print 
    for group in case_groups:
        print ("Constructing " + file_names[group -1].split("_")[1] +  "cohort: " + str(group))
        cohort = buildDataSubset(metadata_df, var, group, control_value, False)
        cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_Cohorts/frequency_cohorts_standard_3/" + var + file_names[group -1])
    print
    print 
    print     

    
    
    
    
    
    
    
    



























