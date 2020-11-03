#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jul  1 15:42:27 2019
@author: Jack G. Sklar
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances



def process_AGP_population(dir_path):
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
    
    
    matching_features = ["longitude", "latitude"]
    cohort_feats = ["sex", "age_years", "bmi", "bmi_cat", "country", "longitude", "latitude", "race", "ibd", "diabetes", "antibiotic_history"]
    
    metadata_matching = metadata_df.loc[:, cohort_feats]
    metadata_matching["antibiotic_history"] = metadata_matching["antibiotic_history"].map({"Year": 0, "I have not taken antibiotics in the past year.": 1, "6 months": 2, "Month": 3, "Week": 4})
    metadata_matching = metadata_matching.loc[:, matching_features]
    scaler = StandardScaler()
    metadata_matching= pd.DataFrame(scaler.fit_transform(metadata_matching.loc[:, matching_features].astype(float)), index = metadata_matching.index, columns = metadata_matching.columns)
    return metadata_df


def process_metadata_population(remove_diseases=False):    
    """
    Metadata preprocessing for cohort construction for Phase I and Phase II 
    """
    feature_groups = pd.read_csv("/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Data/Cleaned_data/feature_groups.csv", index_col = 0)
    metadata_df = pd.read_csv("/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
    feature_groups = feature_groups[~feature_groups.index.isin(['dna_extracted','physical_specimen_remaining','public', 'breastmilk_formula_ensure', 
                                                                'acne_medication', 'acne_medication_otc', 'alcohol_consumption'])]
    binary_features = feature_groups[feature_groups["group"] == "binary"].index.values
    print("Population size (full): ", len(metadata_df))
    
    ### Removal of samples that did not have valid annotation of important information used for matching
    for val in ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]:
        metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
    print("Population size (removed missing metadata samples): ", len(metadata_df))
    
    metadata_df.loc[:, ["bmi", "age_years"]] = metadata_df.loc[:, ["bmi", "age_years"]].astype(float)
    
    ## define set of healthy population for construction of binary questionnaire variables:
    ##      - adults (20:80), with antibiotic use in the last 6-months or later, no IBD or Diabetes, excluding obese participants
    ##      - only include most represented countries (US, UK, Can)
    metadata_df = metadata_df[(metadata_df["age_years"] >= 20.0) & (metadata_df["age_years"] <= 80.0) &
                              (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                              (metadata_df["ibd"] == 0) &
                              (metadata_df["diabetes"] == 0) &
                              (metadata_df["bmi"] >= 12.5) & (metadata_df["bmi"] <= 40.0) &
                              (metadata_df["country"].isin(["USA", "United Kingdom", "Canada"]))]
    print("Population size (removed standard exclusion): ", len(metadata_df))
    if remove_diseases:
        metadata_df = metadata_df[(metadata_df["acid_reflux"] != 1) & (metadata_df["add_adhd"] != 1) & (metadata_df["asd"] != 1) & (metadata_df["autoimmune"] != 1) &
                                  (metadata_df["cancer"] != 1) & (metadata_df["cardiovascular_disease"] != 1) & (metadata_df["gluten"] != "I was diagnosed with celiac disease") &
                                  (metadata_df["depression_bipolar_schizophrenia"] != 1) & (metadata_df["fungal_overgrowth"] != 1) &
                                  (metadata_df["ibd"] != 1) & (metadata_df["ibs"] != 1) & (metadata_df["liver_disease"] != 1) & (metadata_df["lung_disease"] != 1) &
                                  (metadata_df["mental_illness"] != 1) & (metadata_df["mental_illness_type_depression"] != 1) & (metadata_df["migraine"] != 1) &
                                  (metadata_df["sibo"] != 1) & (metadata_df["skin_condition"] != 1) & (metadata_df["thyroid"] != 1) & (metadata_df["kidney_disease"] != 1) &
                                  (metadata_df["asd"] != 1) & (metadata_df["ibd"] != 1) & (metadata_df["cdiff"] != 1) &
                                  (metadata_df["mental_illness_type_ptsd_posttraumatic_stress_disorder"] != 1) & (metadata_df["skin_condition"] != 1) &
                                  (metadata_df["alzheimers"] != 1) & (metadata_df["epilepsy_or_seizure_disorder"] != 1) & (metadata_df["pku"] != 1)]
        print("Population size (removed diseased samples): ", len(metadata_df))
    
    metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].replace(["I tend to have diarrhea (watery stool)", "I tend to have diarrhea (watery stool) - Type 5, 6 and 7"], "loose")
    metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].replace(["I tend to have normal formed stool", "I tend to have normal formed stool - Type 3 and 4"], "normal")
    metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].replace(["I tend to be constipated (have difficulty passing stool)", "I tend to be constipated (have difficulty passing stool) - Type 1 and 2"], "hard")
    
    ### preprocess host metadata used for matching: encode as integers to calculate cosine similarity matrix
    metadata_df["race"] = metadata_df["race"].replace(["Unspecified", "Not provided"], "Other")
    metadata_df['sex'] = metadata_df['sex'].replace(["Unspecified", "Not provided", "unspecified"], "Other")
    metadata_df["diet_type"] = metadata_df["diet_type"].replace(["Unspecified", "Not provided"], "Other")
    metadata_df["longitude"] = metadata_df["longitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)
    metadata_df["latitude"] = metadata_df["latitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)
    le_race = LabelEncoder()
    le_sex = LabelEncoder()
    le_sex.fit(metadata_df["sex"].unique())
    metadata_df["sex"] = le_sex.transform(metadata_df["sex"])
    le_race.fit(metadata_df["race"].unique())
    metadata_df["race"] = le_race.transform(metadata_df["race"])
    metadata_df["diet_type"] = metadata_df["diet_type"].map({"Omnivore": 0,
                                                             "Omnivore but do not eat red meat": 1,
                                                             "Vegetarian but eat seafood": 2, 
                                                             "Vegetarian": 3,
                                                             "Vegan": 4,
                                                             "Other": 5})
    metadata_df["antibiotic_history"] = metadata_df["antibiotic_history"].map({"Year": 0, 
                                                                               "I have not taken antibiotics in the past year.": 1})
    ## Samples with missing geographic locations get the centroid of their country or residence
    usa_missing_geo = metadata_df[(metadata_df["longitude"] == -10.0) & (metadata_df["country"] == "USA")].index.values
    uk_missing_geo = metadata_df[(metadata_df["longitude"] == -10.0) & (metadata_df["country"] == "United Kingdom")].index.values
    can_missing_geo = metadata_df[(metadata_df["longitude"] == -10.0) & (metadata_df["country"] == "Canada")].index.values
    metadata_df.loc[usa_missing_geo, ["longitude", "latitude"]] = (-98.6, 39.8)
    metadata_df.loc[uk_missing_geo, ["longitude", "latitude"]] = (-1.5, 52.6)
    metadata_df.loc[can_missing_geo, ["longitude", "latitude"]] = (-79.4, 43.9)
    return metadata_df, binary_features



def pairIDs(length):
    ## annotate Adjacent pairs in the cohort dataframe as matched paris
    num_pairs = int(length/2)
    pair_ids = []
    for val in range(num_pairs):
        pair_ids.append(val)
        pair_ids.append(val)
    return pair_ids


def buildDataSubset(metadata_df, target_var, pos_target, neg_target):
    """    
    Compute cosine similarity matrix, for each positive sample, 
    choose the most similar negative sample from the control population
    sample without replacement, stop once each positive sample is matched
    to a control maintaining a balanced cohort
        pos_target: positive class label
        neg_target: negavtive class label (possibly a list of labels)
        target_var: which frequency variable to make a balanced cohort for

    """
    target = metadata_df[target_var]
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
    matching_features = ["longitude", "latitude"]
    cohort_feats = ["sex", "age_years", "bmi", "bmi_cat", "country", "longitude", 
                    "latitude", "race", "antibiotic_history", "diet_type",  
                    "alcohol_frequency","milk_cheese_frequency", "meat_eggs_frequency", 
                    "bowel_movement_quality", "vegetable_frequency"]
    # Standard Scale all feautres so they have equal impact on the cosine similarity 
    # calculations between positive class (cases), and the healthy control population
    metadata_matching = metadata_df.loc[:, matching_features]
    scaler = StandardScaler()
    metadata_matching= pd.DataFrame(scaler.fit_transform(metadata_df.loc[:, matching_features].astype(float)), 
                                    index = metadata_df.index, columns = matching_features)
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
    if pos_target != 1:
        cohort["target"] = cohort["target"].map({pos_target: 1, neg_target: 0})
    cohort["pairID"] = pairIDs(len(cohort))
    cohort["pairDist"] = distances
    cohort["worstPairDist"] = worst_distances
    return cohort
