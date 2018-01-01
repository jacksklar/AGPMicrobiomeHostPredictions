#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:22:24 2019

@author: jacksklar
"""
#import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler


metadata_df = pd.read_csv("/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
for val in ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]


metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].replace(["I tend to have diarrhea (watery stool)", "I tend to have diarrhea (watery stool) - Type 5, 6 and 7"], 0)
metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].replace(["I tend to have normal formed stool", "I tend to have normal formed stool - Type 3 and 4"], 1)
metadata_df["bowel_movement_quality"] = metadata_df["bowel_movement_quality"].replace(["I tend to be constipated (have difficulty passing stool)", "I tend to be constipated (have difficulty passing stool) - Type 1 and 2"], 2)

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

cohort_matching_features = ["longitude", "latitude"]
cohort_feats = ["sex", "age_years", "bmi", "bmi_cat", "country", "longitude", "latitude", "race", 
                "antibiotic_history", "diet_type", "alcohol_frequency","milk_cheese_frequency", 
                "bowel_movement_quality", "meat_eggs_frequency", "vegetable_frequency"]


metadata_matching = metadata_df.loc[:, cohort_matching_features]
scaler = StandardScaler() 
metadata_matching= pd.DataFrame(scaler.fit_transform(metadata_matching.loc[:, cohort_matching_features].astype(float)), index = metadata_matching.index, columns = metadata_matching.columns)

def pairIDs(length):
    num_pairs = int(length/2)
    pair_ids = []
    for val in range(num_pairs):
        pair_ids.append(val)
        pair_ids.append(val)
    return pair_ids


def buildDataSubset(metadata, target_var, pos_target, neg_target, match_subset): 
    ##target_var: name of metadata variable
    ##pos_target: positive class label
    ##neg_target: negavtive class label (possibly a list of labels)
    ##age_related: boolean (remove age related variables from matchign criteria)
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

save_path = "/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Feature_Cohorts/Phase_II_Cohorts/binary_cohorts/"


#%%
"""
disease_list = ["acid_reflux", "add_adhd", "asd", "autoimmune", "cancer", "cardiovascular_disease", 
                "depression_bipolar_schizophrenia", "fungal_overgrowth", "ibd", "ibs", "liver_disease",
                "lung_disease", "mental_illness", "mental_illness_type_depression", "migraine", "sibo", 
                "skin_condition", "thyroid", "asd", "ibd", "cdiff", "mental_illness_type_ptsd_posttraumatic_stress_disorder", 
                "skin_condition", "alzheimers", "epilepsy_or_seizure_disorder", "pku"]
"""
##Age
metadata_df_healthy = metadata_df[(metadata_df["ibd"] == 0) &
                                  (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                                  (metadata_df["diabetes"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 40.0)]
#for disease in disease_list:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]

print ("Constructing child [6:18] cohort with matched healthy adults")
metadata_df_healthy_young = metadata_df_healthy[metadata_df_healthy["age_years"] <= 80.0]
metadata_df_healthy_young["age_group"] = metadata_df_healthy_young["age_years"] 
young = metadata_df_healthy_young[(metadata_df_healthy_young["age_years"] <= 18.0) & (metadata_df_healthy_young["age_years"] >= 6.0)].index
old = metadata_df_healthy_young[metadata_df_healthy_young["age_years"] >= 20.0].index
metadata_df_healthy_young.loc[young, "age_group"] = 1
metadata_df_healthy_young.loc[old, "age_group"] = 0
young_cohort = buildDataSubset(metadata_df_healthy_young, "age_group", 1, 0, None)
if len(young_cohort) >= 80:
    young_cohort.to_csv(save_path + "age_6-18.csv")

print ("Constructing toddler [3:5] cohort with matched healthy adults")
metadata_df_healthy_toddler = metadata_df_healthy[metadata_df_healthy["age_years"] <= 80.0]
metadata_df_healthy_toddler["age_group"] = metadata_df_healthy_toddler["age_years"] 
infant = metadata_df_healthy_toddler[(metadata_df_healthy_toddler["age_years"] <= 5.0) & (metadata_df_healthy_toddler["age_years"] >= 3.0)].index
old = metadata_df_healthy_toddler[metadata_df_healthy_toddler["age_years"] >= 20.0].index
metadata_df_healthy_toddler.loc[infant, "age_group"] = 1
metadata_df_healthy_toddler.loc[old, "age_group"] = 0
infant_cohort = buildDataSubset(metadata_df_healthy_toddler, "age_group", 1, 0, None)
if len(infant_cohort) >= 80:
    infant_cohort.to_csv(save_path + "age_3-5.csv")

print ("Constructing 2 and under cohort with matched healthy adults")
metadata_df_healthy_infant = metadata_df_healthy[metadata_df_healthy["age_years"] <= 80.0]
metadata_df_healthy_infant["age_group"] = metadata_df_healthy_infant["age_years"] 
infant = metadata_df_healthy_infant[metadata_df_healthy_infant["age_years"] <= 2.0].index
old = metadata_df_healthy_infant[metadata_df_healthy_infant["age_years"] >= 20.0].index
metadata_df_healthy_infant.loc[infant, "age_group"] = 1
metadata_df_healthy_infant.loc[old, "age_group"] = 0
infant_cohort = buildDataSubset(metadata_df_healthy_infant, "age_group", 1, 0, None)
if len(infant_cohort) >= 80:
    infant_cohort.to_csv(save_path + "age_1-2.csv")

print ("Constructing 65 and over cohort with matched healthy adults")
metadata_df_healthy_old = metadata_df_healthy[metadata_df_healthy["age_years"] >= 20.0]
metadata_df_healthy_old["age_group"] = metadata_df_healthy_old["age_years"] 
young = metadata_df_healthy_old[metadata_df_healthy_old["age_years"] < 70.0].index
old = metadata_df_healthy_old[metadata_df_healthy_old["age_years"] >= 70.0].index
metadata_df_healthy_old.loc[young, "age_group"] = 0
metadata_df_healthy_old.loc[old, "age_group"] = 1
old_cohort = buildDataSubset(metadata_df_healthy_old, "age_group", 1, 0, None)
if len(old_cohort) >= 80:
    old_cohort.to_csv(save_path + "age_over70.csv")


###BMI SPECIAL FEATURE:
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                  (metadata_df["age_years"] <= 80.0) &
                                  (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                                  (metadata_df["ibd"] == 0) &
                                  (metadata_df["diabetes"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 40.0)]
#for disease in disease_list:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]

print ("Constructing Obese cohort with matched normal samples")
obese_cohort = buildDataSubset(metadata_df_healthy, "bmi_cat", "Obese", "Normal", None)
if len(obese_cohort) >= 80:
    obese_cohort.to_csv(save_path + "Obese_cohort.csv")

print ("Constructing Overweight cohort with matched normal samples")
overweight_cohort = buildDataSubset(metadata_df_healthy, "bmi_cat", "Overweight", "Normal", None)
if len(overweight_cohort) >= 80:
    overweight_cohort.to_csv(save_path + "Overweight_cohort.csv")

print ("Constructing Underweight cohort with matched normal samples")
underweight_cohort = buildDataSubset(metadata_df_healthy, "bmi_cat", "Underweight", "Normal", None)
if len(underweight_cohort) >= 80:
    underweight_cohort.to_csv(save_path + "Underweight_cohort.csv")


###IBD
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                  (metadata_df["age_years"] <= 80.0) &
                                  (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                                  (metadata_df["diabetes"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 40.0)]

#disease_list_ibd  = list(set(disease_list) - set(["ibd"]))
#for disease in disease_list_ibd:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]
    
print ("Constructing IBD cohort with matched healthy samples")
ibd_cohort = buildDataSubset(metadata_df_healthy, "ibd", 1, 0, None)
if len(ibd_cohort) >= 80:
    ibd_cohort.to_csv(save_path + "IBD_cohort.csv")

##Antibiotic History
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                  (metadata_df["age_years"] <= 80.0) &
                                  (metadata_df["ibd"] == 0) &
                                  (metadata_df["diabetes"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 40.0)]
#for disease in disease_list:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]
    
print ("Constructing Antiobiotic Year cohort with matched healthy samples")
antiB_cohort1 = buildDataSubset(metadata_df_healthy, "antibiotic_history", "Year", "I have not taken antibiotics in the past year.", None)
if len(antiB_cohort1) >= 80:
    antiB_cohort1.to_csv(save_path + "antiB_Year_cohort.csv")

print ("Constructing Antiobiotic 6 Months cohort with matched healthy samples")
antiB_cohort2 = buildDataSubset(metadata_df_healthy, "antibiotic_history", "6 months", "I have not taken antibiotics in the past year.", None)
if len(antiB_cohort2) >= 80:
    antiB_cohort2.to_csv(save_path + "antiB_6Month_cohort.csv")

print ("Constructing Antiobiotic Year cohort with matched healthy samples")
antiB_cohort3 = buildDataSubset(metadata_df_healthy, "antibiotic_history", "Month", "I have not taken antibiotics in the past year.",  None)
if len(antiB_cohort3) >= 80:
    antiB_cohort3.to_csv(save_path + "antiB_Month_cohort.csv")

print ("Constructing Antiobiotic Week cohort with matched healthy samples")
antiB_cohort4 = buildDataSubset(metadata_df_healthy, "antibiotic_history", "Week", "I have not taken antibiotics in the past year.", None)
if len(antiB_cohort4) >= 80:
    antiB_cohort4.to_csv(save_path + "antiB_Week_cohort.csv")

##Diabetes
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                  (metadata_df["age_years"] <= 80.0) &
                                  (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                                  (metadata_df["ibd"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 60.0)]
#disease_list_ibd  = list(set(disease_list) - set(["diabetes"]))
#for disease in disease_list_ibd:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]

print ("Constructing Diabetes type II cohort with matched healthy samples")
metadata_df_healthy.loc[metadata_df_healthy["diabetes_type"].isin(["Not provided", "Unspecified"]), "diabetes_type"] = 0
diabetes_cohort = buildDataSubset(metadata_df_healthy, "diabetes_type", "Type II diabetes", 0, None)
if len(diabetes_cohort) >= 80:
    diabetes_cohort.to_csv(save_path + "diabetes_typeII_cohort.csv")

##Gender
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                  (metadata_df["age_years"] <= 80.0) &
                                  (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                                  (metadata_df["ibd"] == 0) &
                                  (metadata_df["sex"] != "other") &
                                  (metadata_df["diabetes"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 40.0)]
#for disease in disease_list:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]
    
gender_healthy_cohort = buildDataSubset(metadata_df_healthy, "sex", "male", "female", None)
if len(gender_healthy_cohort) >= 80:
    gender_healthy_cohort.to_csv(save_path + "gender_healthy_cohort.csv")

###Contraceptive (just women)
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                  (metadata_df["age_years"] <= 80.0) &
                                  (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                                  (metadata_df["ibd"] == 0) &
                                  (metadata_df["sex"] == "female") &
                                  (metadata_df["diabetes"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 40.0)]
#for disease in disease_list:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]
    
print ("Constructing contraceptive pill cohort with matched healthy samples")
pill_cohort = buildDataSubset(metadata_df_healthy, "contraceptive", "Yes, I am taking the \"pill\"", "No", None)
if len(pill_cohort) >= 80:
    pill_cohort.to_csv(save_path + "pill_cohort.csv")

print ("Constructing IUD cohort with matched healthy samples")
iud_cohort = buildDataSubset(metadata_df_healthy, "contraceptive", "Yes, I use a hormonal IUD (Mirena)", "No", None)
if len(iud_cohort) >= 80:
    iud_cohort.to_csv(save_path + "iud_cohort.csv")

####Autism: (just children)
metadata_df_healthy = metadata_df[(metadata_df["age_years"] <= 30.0) &
                                  (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                                  (metadata_df["ibd"] == 0) &
                                  (metadata_df["diabetes"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 40.0)]

#disease_list_ibd  = list(set(disease_list) - set(["asd"]))
#for disease in disease_list_ibd:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]
    
print ("Constructing Autism cohort with matched healthy samples (under 30)")
asd_cohort = buildDataSubset(metadata_df_healthy, "asd", 1, 0, None)
if len(asd_cohort) >= 80:
    asd_cohort.to_csv(save_path + "asd_cohort.csv")

####Standard Exclusion Criteria 
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                  (metadata_df["age_years"] <= 80.0) &
                                  (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                                  (metadata_df["ibd"] == 0) &
                                  (metadata_df["diabetes"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 40.0)]
#for disease in disease_list:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]

alc_cohort = buildDataSubset(metadata_df_healthy, "drinks_per_session", "1", "I don't drink", None)
if len(alc_cohort) >= 80:
    alc_cohort.to_csv(save_path + "drinks_1_cohort.csv")

alc_cohort = buildDataSubset(metadata_df_healthy, "drinks_per_session", "1-2", "I don't drink", None)
if len(alc_cohort) >= 80:
    alc_cohort.to_csv(save_path + "drinks_1_2_cohort.csv")

alc_cohort = buildDataSubset(metadata_df_healthy, "drinks_per_session", "2-3", "I don't drink", None)
if len(alc_cohort) >= 80:
    alc_cohort.to_csv(save_path + "drinks_2_3_cohort.csv")

alc_cohort = buildDataSubset(metadata_df_healthy, "drinks_per_session", "3-4", "I don't drink", None)
if len(alc_cohort) >= 80:
    alc_cohort.to_csv(save_path + "drinks_3_4_cohort.csv")

alc_cohort = buildDataSubset(metadata_df_healthy, "drinks_per_session", "4+", "I don't drink", None)
if len(alc_cohort) >= 80:
    alc_cohort.to_csv(save_path + "drinks_4_plus_cohort.csv")

###Alcohol_Consumption Frequeny:
print ("Constructing alcohol daily with matched healthy samples")
alc_cohort = buildDataSubset(metadata_df_healthy, "alcohol_frequency", 4, 0, None)
if len(alc_cohort) >= 80:
    alc_cohort.to_csv(save_path + "alcohol_daily_cohort.csv")

print ("Constructing alcohol rare cohort with matched healthy samples")
alc_cohort = buildDataSubset(metadata_df_healthy, "alcohol_frequency", 1, 0, None)
if len(alc_cohort) >= 80:
    alc_cohort.to_csv(save_path + "alcohol_rare_cohort.csv")

print ("Constructing alcohol occasional cohort with matched healthy samples")
alc_cohort = buildDataSubset(metadata_df_healthy, "alcohol_frequency", 2, 0, None)
if len(alc_cohort) >= 80:
    alc_cohort.to_csv(save_path + "alcohol_occasional_cohort.csv")

print ("Constructing alcohol regular cohort with matched healthy samples")
alc_cohort = buildDataSubset(metadata_df_healthy, "alcohol_frequency", 3, 0, None)
if len(alc_cohort) >= 80:
    alc_cohort.to_csv(save_path + "alcohol_regular_cohort.csv")

print ("Constructing plants cohort with matched normal samples")
nplant_6_10_cohort = buildDataSubset(metadata_df_healthy, "types_of_plants", "6 to 10", "Less than 5", None)
if len(nplant_6_10_cohort) >= 80:
    nplant_6_10_cohort.to_csv(save_path + "nplant_6_10_cohort.csv")

print ("Constructing plants cohort with matched normal samples")
nplant_11_20_cohort = buildDataSubset(metadata_df_healthy, "types_of_plants", "11 to 20", "Less than 5", None)
if len(nplant_11_20_cohort) >= 80:
    nplant_11_20_cohort.to_csv(save_path + "nplant_11_20_cohort.csv")

print ("Constructing plants cohort with matched normal samples")
nplant_21_30_cohort = buildDataSubset(metadata_df_healthy, "types_of_plants", "21 to 30", "Less than 5", None)
if len(nplant_21_30_cohort) >= 80:
    nplant_21_30_cohort.to_csv(save_path + "nplant_21_30_cohort.csv")

print ("Constructing plants cohort with matched normal samples")
nplant_30_plus_cohort = buildDataSubset(metadata_df_healthy, "types_of_plants", "More than 30", "Less than 5", None)
if len(nplant_30_plus_cohort) >= 80:
    nplant_30_plus_cohort.to_csv(save_path + "nplant_30_plus_cohort.csv")

print ("Constructing no gluten cohort with matched healthy samples")
no_gluten_cohort = buildDataSubset(metadata_df_healthy, "gluten", "I do not eat gluten because it makes me feel bad", "No", None)
if len(no_gluten_cohort) >= 80:
    no_gluten_cohort.to_csv(save_path + "no_gluten_cohort.csv")

print ("Constructing celiac cohort with matched healthy samples")
celiac_cohort = buildDataSubset(metadata_df_healthy, "gluten", "I was diagnosed with celiac disease", "No", None)
if len(celiac_cohort) >= 80:
    celiac_cohort.to_csv(save_path + "celiac_cohort.csv")

print ("Constructing gluten_allergy cohort with matched healthy samples")
gluten_alergy_cohort = buildDataSubset(metadata_df_healthy, "gluten", "I was diagnosed with gluten allergy (anti-gluten IgG), but not celiac disease", "No", None)
if len(gluten_alergy_cohort) >= 80:
    gluten_alergy_cohort.to_csv(save_path + "gluten_alergy_cohort.csv")

print ("Constructing USA immigrant cohort with matched healthy samples")
metadata_df_healthy["immigrant"] = metadata_df_healthy["country_of_birth"] 
immigrants_us = metadata_df_healthy[(metadata_df_healthy["country_of_birth"] != "United States") & (metadata_df_healthy["country_residence"] == "United States")].index
native_us = metadata_df_healthy[(metadata_df_healthy["country_of_birth"] == "United States") & (metadata_df_healthy["country_residence"] == "United States")].index
metadata_df_healthy.loc[immigrants_us, "immigrant"] = 1
metadata_df_healthy.loc[native_us, "immigrant"] = 0
usa_immigrant_cohort = buildDataSubset(metadata_df_healthy, "immigrant", 1, 0, None)
if len(usa_immigrant_cohort) >= 80:
    usa_immigrant_cohort.to_csv(save_path + "usa_immigrant_cohort.csv")

print ("Constructing UK immigrant cohort with matched healthy samples")
metadata_df_healthy["immigrant"] = metadata_df_healthy["country_of_birth"] 
immigrants_uk = metadata_df_healthy[(metadata_df_healthy["country_of_birth"] != "United Kingdom") & (metadata_df_healthy["country_residence"] == "United Kingdom")].index
native_uk = metadata_df_healthy[(metadata_df_healthy["country_of_birth"] == "United Kingdom") & (metadata_df_healthy["country_residence"] == "United Kingdom")].index
metadata_df_healthy.loc[immigrants_uk, "immigrant"] = 1
metadata_df_healthy.loc[native_uk, "immigrant"] = 0
uk_immigrant_cohort = buildDataSubset(metadata_df_healthy, "immigrant", 1, 0, None)
if len(uk_immigrant_cohort) >= 80:
    uk_immigrant_cohort.to_csv(save_path + "uk_immigrant_cohort.csv")

print ("Constructing diet:  cohort with matched healthy samples")
omnivore_noRed_cohort = buildDataSubset(metadata_df_healthy, "diet_type", "Omnivore but do not eat red meat", "Omnivore", None)
if len(omnivore_noRed_cohort) >= 80:
    omnivore_noRed_cohort.to_csv(save_path + "omnivore_noRed_cohort.csv")

print ("Constructing diet: Pescatarian cohort with matched healthy samples")
pescatarian_cohort = buildDataSubset(metadata_df_healthy, "diet_type", "Vegetarian but eat seafood", "Omnivore", None)
if len(pescatarian_cohort) >= 80:
    pescatarian_cohort.to_csv(save_path + "pescatarian_cohort.csv")

print ("Constructing diet: Vegetarian cohort with matched healthy samples")
vegetarian_cohort = buildDataSubset(metadata_df_healthy, "diet_type", "Vegetarian", "Omnivore", None)
if len(vegetarian_cohort) >= 80:
    vegetarian_cohort.to_csv(save_path + "vegetarian_cohort.csv")

print ("Constructing diet: Vegan cohort with matched healthy samples")
vegan_cohort = buildDataSubset(metadata_df_healthy, "diet_type", "Vegan", "Omnivore", None)
if len(vegan_cohort) >= 80:
    vegan_cohort.to_csv(save_path + "vegan_cohort.csv")

print ("Constructing BMF 1 cohort with matched healthy samples")
bmf_less1_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Less than one", "One", None)
if len(bmf_less1_cohort) >= 80:
    bmf_less1_cohort.to_csv(save_path + "bmf_less1_cohort.csv")

print ("Constructing BMF 2 cohort with matched healthy samples")
bmf_2_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Two", "One", None)
if len(bmf_2_cohort) >= 80:
    bmf_2_cohort.to_csv(save_path + "bmf_2_cohort.csv")

print ("Constructing BMF 3 cohort with matched healthy samples")
bmf_3_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Three", "One", None)
if len(bmf_3_cohort) >= 80:
    bmf_3_cohort.to_csv(save_path + "bmf_3_cohort.csv")

print ("Constructing BMF 4 cohort with matched healthy samples")
bmf_4_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Four", "One", None)
if len(bmf_4_cohort) >= 80:
    bmf_4_cohort.to_csv(save_path + "bmf_4_cohort.csv")

print ("Constructing BMF 5 cohort with matched healthy samples")
bmf_5_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Five or more", "One", None)
if len(bmf_5_cohort) >= 80:
    bmf_5_cohort.to_csv(save_path + "bmf_5_cohort.csv")

print ("Constructing BMQ Solid cohort with matched healthy samples")
bmq_solid_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_quality", 2, 1, None)
if len(bmq_solid_cohort) >= 80:
    bmq_solid_cohort.to_csv(save_path + "bmq_solid_cohort.csv")

print ("Constructing BMQ loose cohort with matched healthy samples")
bmq_loose_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_quality", 0, 1, None)
if len(bmq_loose_cohort) >= 80:
    bmq_loose_cohort.to_csv(save_path + "bmq_loose_cohort.csv")

print ("Constructing weight increase cohort with matched healthy samples")
weigth_increase_cohort = buildDataSubset(metadata_df_healthy, "weight_change", "Increased more than 10 pounds", "Remained stable", None)
if len(weigth_increase_cohort) >= 80:
    weigth_increase_cohort.to_csv(save_path + "weigth_increase_cohort.csv")

print ("Constructing weight decrease cohort with matched healthy samples")
weigth_decrease_cohort = buildDataSubset(metadata_df_healthy, "weight_change", "Decreased more than 10 pounds", "Remained stable", None)
if len(weigth_decrease_cohort) >= 80:
    weigth_decrease_cohort.to_csv(save_path + "weigth_decrease_cohort.csv")

print ("Constructing country cohort with matched healthy samples")
country_cohort = buildDataSubset(metadata_df_healthy, "country", "USA", "United Kingdom", None)
if len(country_cohort) >= 80:
    country_cohort.to_csv(save_path + "country_cohort.csv")

metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                  (metadata_df["age_years"] <= 80.0) &
                                  (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                                  (metadata_df["ibd"] == 0) &
                                  (metadata_df["diabetes"] == 0) &
                                  (metadata_df["bmi"] >= 12.5) & 
                                  (metadata_df["bmi"] <= 40.0)]
#or disease in disease_list:
#    metadata_df_healthy = metadata_df_healthy[metadata_df_healthy[disease] != 1]
    
metadata_df_healthy = metadata_df_healthy[~metadata_df_healthy["drinks_per_session"].isin(["Unspecified", "2-Jan", "3-Feb", "Not provided"])]
metadata_df_healthy["drinks_per_session"] = metadata_df_healthy["drinks_per_session"].map({"I don't drink": 0, "1": 1, "1-2": 2, "2-3": 3,"3-4": 4,"4+": 5})
metadata_df_healthy["average_drinks_per_week"] = metadata_df_healthy["drinks_per_session"].astype(str) + "_" + metadata_df_healthy["alcohol_frequency"].astype(str)

drink_map = {'0_0': "non_drinker", '0_1': "0", '0_2': "0", '0_4': "0",
             '1_0': "0", '1_1': "light_drinker", '1_2': "light_drinker", '1_3': "mild_drinker", 
             '1_4': "mild_drinker", '2_0': "0", '2_1': "light_drinker", '2_2': "light_drinker", 
             '2_3': "mild_drinker", '2_4': "heavy_drinker", '3_0': "0", '3_1': "light_drinker", 
             '3_2': "mild_drinker", '3_3': "heavy_drinker", '3_4': "heavy_drinker", '4_1': "light_drinker", 
             '4_2': "mild_drinker", '4_3': "heavy_drinker", '4_4': "heavy_drinker", '5_1': "light_drinker", 
             '5_2': "mild_drinker", '5_3': "heavy_drinker", '5_4': "heavy_drinker"}  
metadata_df_healthy["average_drinks_per_week"] = metadata_df_healthy["average_drinks_per_week"].map(drink_map)

drinker_type_cohort = buildDataSubset(metadata_df_healthy, "average_drinks_per_week", "light_drinker", "non_drinker", None)
if len(drinker_type_cohort) >= 80:
    drinker_type_cohort.to_csv(save_path + "light_drinker_type_cohort.csv")

drinker_type_cohort = buildDataSubset(metadata_df_healthy, "average_drinks_per_week", "mild_drinker", "non_drinker", None)
if len(drinker_type_cohort) >= 80:
    drinker_type_cohort.to_csv(save_path + "mild_drinker_type_cohort.csv")

drinker_type_cohort = buildDataSubset(metadata_df_healthy, "average_drinks_per_week", "heavy_drinker", "non_drinker", None)
if len(drinker_type_cohort) >= 80:
    drinker_type_cohort.to_csv(save_path + "heavy_drinker_type_cohort.csv")


