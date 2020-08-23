#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 23 13:51:58 2019
@author: Jack G. Sklar
"""

from cohort_construction_utils import buildDataSubset, process_metadata_population



def construct_binary_cohorts(output_path, exclude_diseases=False):
    metadata_df, binary_features = process_metadata_population(remove_diseases=exclude_diseases)
    for feature in binary_features:
        feature_counts = metadata_df[feature].value_counts()
        if 1 not in feature_counts.index:    ## Skip binary variables with only one value (e.x. sampling location)
            continue
        n_feature_positive = feature_counts[1]
        if n_feature_positive < 40:    ## Skip variables with less than 50 positive samples
            print("       Not enough samples", feature, str(n_feature_positive))
            continue
        else:
            print(feature, str(n_feature_positive))
            cohort = buildDataSubset(metadata_df, feature, 1, 0)
            cohort.to_csv(output_path + str(feature) + ".csv")
            
        
        
output_path = "/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Feature_Cohorts/Phase_II_Cohorts/binary_cohorts/"
construct_binary_cohorts(output_path, exclude_diseases=False)


output_path = "/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Feature_Cohorts/Phase_II_Cohorts_disease_removed/binary_cohorts/"
construct_binary_cohorts(output_path, exclude_diseases=True)