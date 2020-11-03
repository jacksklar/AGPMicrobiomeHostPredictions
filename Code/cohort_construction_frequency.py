#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:55:00 2019
@author: Jack G. Sklar
"""

import numpy as np 
from cohort_construction_utils import buildDataSubset, process_metadata_population



def create_frequency_cohorts(output_path, exclude_diseases=False): 
    metadata_df, _ = process_metadata_population(remove_diseases=exclude_diseases)
    freq_values = ['artificial_sweeteners','exercise_frequency','fermented_plant_frequency','frozen_dessert_frequency',
                   'fruit_frequency', 'high_fat_red_meat_frequency','homecooked_meals_frequency','meat_eggs_frequency','milk_cheese_frequency',
                   'milk_substitute_frequency','olive_oil','one_liter_of_water_a_day_frequency','poultry_frequency','prepared_meals_frequency',
                   'probiotic_frequency','ready_to_eat_meals_frequency','red_meat_frequency', 'salted_snacks_frequency', 'seafood_frequency',
                   'smoking_frequency', 'sugar_sweetened_drink_frequency', 'sugary_sweets_frequency',
                   'vegetable_frequency','vitamin_b_supplement_frequency','vitamin_d_supplement_frequency','whole_eggs',
                   'whole_grain_frequency','alcohol_frequency']
    file_names = ["_rare_cohort.csv", "_occasional_cohort.csv", "_regular_cohort.csv", "_daily_cohort.csv"]
    combine_threshold = 100 if not exclude_diseases else 100
    for var in freq_values:
        group_counts = metadata_df[var].value_counts()
        print(group_counts)
        # combine never group if size is bellow threshold and combine with occasionally if still too small
        if group_counts[0] <= combine_threshold:
            print("      Never group: ",  group_counts[0], group_counts[0] + group_counts[1])
            metadata_df.loc[metadata_df[var] == 1, var] = 0
            group_counts = metadata_df[var].value_counts()
            if group_counts[0] <= combine_threshold:
                print("      Never group: ",  group_counts[0], group_counts[0] + group_counts[2])
                metadata_df.loc[metadata_df[var] == 2, var] = 0
        # combine daily group if size is bellow threshold and combine with occasionally if still too small
        if group_counts[4] <= combine_threshold:
            print("      Daily group: ",  group_counts[4], group_counts[4] + group_counts[3])
            metadata_df.loc[metadata_df[var] == 3, var] = 4
            group_counts = metadata_df[var].value_counts()
            if group_counts[4] <= combine_threshold:
                print("      Daily group: ",  group_counts[4], group_counts[4] + group_counts[2])
                metadata_df.loc[metadata_df[var] == 2, var] = 4
    for var in freq_values:
        print(var)
        frequency_groups = np.sort(metadata_df[var].unique())
        if 5 in frequency_groups:
            frequency_groups = np.delete(frequency_groups, np.argwhere(frequency_groups == 5))
        control_value = frequency_groups[0]
        case_groups = frequency_groups[1:]
        for group in case_groups:
            if metadata_df[var].value_counts()[group] < 40:
                print("Not constructing " + str(file_names[group -1].split("_")[1]) + " too small")
                continue
            print("Constructing " + file_names[group -1].split("_")[1] +  "cohort: " + str(group))
            cohort = buildDataSubset(metadata_df, var, group, control_value)
            cohort.to_csv(output_path + var + file_names[group -1])



output_path = "/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Feature_Cohorts/Phase_II_Cohorts_disease_removed/frequency_cohorts/"
create_frequency_cohorts(output_path, exclude_diseases=True)

output_path = "/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/Feature_Cohorts/Phase_II_Cohorts/frequency_cohorts/"
create_frequency_cohorts(output_path, exclude_diseases=False)
