#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:10:03 2019

@author: sklarjg
"""

import pandas as pd 
import numpy as np

metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)
frequency_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/frequency_feature_info.csv", index_col = 0)

metadata_df.drop(["age_cat","bmi_corrected","age_corrected","survey_id","subset_healthy",
                  "country_residence","country_of_birth","state","sleep_duration",
                  "bmi_cat", "acne_medication","acne_medication_otc","cancer_treatment",
                  "fermented_plant_frequency.1","census_region","fed_as_infant", "elevation",
                  "contraceptive","flu_vaccine_date","gluten","diabetes_type","level_of_education","weight_change"], axis = 1, inplace = True)

binary_cols = [val for val in feature_info.index]
frequency_cols = [val for val in frequency_info['Variable'].unique()]
full_cols = [val for val in metadata_df.columns]

keep_features = binary_cols + frequency_cols
print "Binary Features" 
for val in full_cols[:54]:
    print val
    metadata_df.loc[metadata_df[val].isin([2]), val] = np.nan
print 
print
print "Frequency Features"

for val in full_cols[54:87]:
    print val
    metadata_df.loc[metadata_df[val].isin([5]), val] = np.nan
print 
print
print "Disease Features"
for val in full_cols[87:110]:
    print val
    metadata_df.loc[metadata_df[val].isin([2,3,4]), val] = np.nan
print 
print
print "Continuous Features" 
for val in ["age_years","bmi","weight_kg","longitude","latitude"]:
    print val
    metadata_df.loc[metadata_df[val].isin(["Unspecified", "Not provided"]), val] = np.nan
    
    

antib_map = {'I have not taken antibiotics in the past year.': 0, 
             'Year': 1,
             '6 months': 2, 
             'Month': 3, 
             'Week': 4, 
             'Not provided': np.nan, 
             'Unspecified': np.nan}

sex_map = {'female': 0, 
             'male': 1,
             'other': np.nan,
             'Not provided': np.nan, 
             'Unspecified': np.nan}

race_map = {'Caucasian' : 'white',
            'Asian or Pacific Islander' : 'asian',
            'Other' : np.nan,
            'Hispanic': "hispanic",
            'Not provided': np.nan,
            'African American': "african_amer",
            'Unspecified': np.nan}
    
diet_map = {'Omnivore': 'omnivore',
            'Omnivore but do not eat red meat': 'omnivore_nored',
            'Vegetarian but eat seafood': 'pescatarian',
            'Vegetarian': 'vegetarian',
            'Vegan': 'vegam',
            'Not provided': np.nan,
            'Unspecified': np.nan}

bmq_map = {"I tend to have diarrhea (watery stool) - Type 5, 6 and 7": 0,
           "I tend to have diarrhea (watery stool)": 0,
           "I tend to have normal formed stool - Type 3 and 4": 1,
           "I tend to have normal formed stool": 1,
           "I tend to be constipated (have difficulty passing stool) - Type 1 and 2": 2,
           "I tend to be constipated (have difficulty passing stool)": 2,
           "I don't know, I do not have a point of reference": np.nan,
           "Unspecified": np.nan,
           "Not provided": np.nan}

bmf_map = {"One": 1,
           "Two": 2,
           "Less than one": 0,
           "Three": 3,
           "Four": 4,
           "Five or more": 5,
           "Not provided": np.nan,
           "Unspecified": np.nan}    

metadata_df.loc[:, "antibiotic_history"] = metadata_df.loc[:, "antibiotic_history"].map(antib_map)
metadata_df.loc[:, "sex"] = metadata_df.loc[:, "sex"].map(sex_map)
metadata_df.loc[:, "bowel_movement_frequency"] = metadata_df.loc[:, "bowel_movement_frequency"].map(bmf_map)
metadata_df.loc[:, "bowel_movement_quality"] = metadata_df.loc[:, "bowel_movement_quality"].map(bmq_map)

metadata_df.loc[:, "race"] = metadata_df.loc[:, "race"].map(race_map)
metadata_df.loc[:, "diet_type"] = metadata_df.loc[:, "diet_type"].map(diet_map)
metadata_df.loc[~metadata_df['country'].isin(["USA","United Kingdom","Australia","Canada"]), 'country'] = np.nan
metadata_df = pd.get_dummies(metadata_df, columns = ["race", "diet_type", "country", "collection_season"])

metadata_df.to_csv("/Users/sklarjg/Desktop/confounding_metadata.csv")
