#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:51:58 2019
@author: sklarjg
"""

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from cohort_construction_utils import buildDataSubset, process_metadata_population



def create_disease_cohorts(output_path, metadata_df, exclude_other_diseases=False):
    disease_list = ["acid_reflux", "add_adhd", "asd", "autoimmune", "cancer", "cardiovascular_disease", 
                "depression_bipolar_schizophrenia", "fungal_overgrowth", "ibd", "ibs", "liver_disease",
                "lung_disease", "mental_illness", "mental_illness_type_depression", "migraine", "sibo", 
                "skin_condition", "thyroid", "asd", "ibd", "cdiff", "mental_illness_type_ptsd_posttraumatic_stress_disorder", 
                "skin_condition", "alzheimers", "epilepsy_or_seizure_disorder", "pku"]
    for disease in disease_list:
        if exclude_other_diseases:
            remove_diseases = list(set(disease_list) - set([disease]))
            disease_df = metadata_df.copy(deep=True)
            for remove in remove_diseases:
                disease_df = disease_df[disease_df[remove] != 1]
        feature_counts = metadata_df[disease].value_counts()
        if 1 not in feature_counts.index:    ## Skip binary variables with only one value (e.x. sampling location)
            continue
        n_feature_positive = feature_counts[1]
        n_feature_negative = feature_counts[0]
        if disease == "mental_illness_type_ptsd_posttraumatic_stress_disorder":
            continue
        if n_feature_positive < 40:    ## Skip variables with less than 50 positive samples
            print("       Not enough samples", disease, str(n_feature_positive))
            continue
        else:
            print(disease, str(n_feature_positive), str(n_feature_negative))
            cohort = buildDataSubset(disease_df, disease, 1, 0)
            cohort.to_csv(output_path + str(target_name) + ".csv")
            


metadata_df, _ = process_metadata_population(remove_diseases=False)

output_path = "/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Feature_Cohorts/Phase_II_Cohorts/binary_cohorts/"  
create_disease_cohorts(output_path, metadata_df, False)


output_path = "/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Feature_Cohorts/Phase_II_Cohorts_disease_removed/binary_cohorts/"  
create_disease_cohorts(output_path, metadata_df, True)


          