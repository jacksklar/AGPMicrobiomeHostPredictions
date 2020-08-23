#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:42:27 2019

@author: sklarjg
"""

import pandas as pd 
from cohort_construction_utils import buildDataSubset, process_AGP_population


dir_path = "/Users/jacksklar/Desktop/AGPMicrobiomeHostPredictions/"
metadata_df = process_AGP_population()

agp_healthy_population = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                     (metadata_df["age_years"] <= 69.0) &
                                     (metadata_df["antibiotic_history"] == "I have not taken antibiotics in the past year.") &
                                     (metadata_df["ibd"] == 0) &
                                     (metadata_df["diabetes"] == 0) &
                                     (metadata_df["bmi"] >= 18.5) & 
                                     (metadata_df["bmi"] < 30.0)]
print("AGP Healthy Population size: ", len(agp_healthy_population))


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
obese_cohort = buildDataSubset(healthy_population, "bmi_cat", "Obese", "Normal")
obese_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/Obese_cohort.csv")

print("Constructing Overweight cohort with matched normal samples")
overweight_cohort = buildDataSubset(healthy_population, "bmi_cat", "Overweight", "Normal")
overweight_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/Overweight_cohort.csv")

print("Constructing Underweight cohort with matched normal samples")
underweight_cohort = buildDataSubset(healthy_population, "bmi_cat", "Underweight", "Normal")
underweight_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/Underweight_cohort.csv")


###IBD
healthy_population = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                     (metadata_df["age_years"] <= 69.0) &
                                     (metadata_df["antibiotic_history"] == "I have not taken antibiotics in the past year.") &
                                     (metadata_df["diabetes"] == 0) &
                                     (metadata_df["bmi"] >= 18.5) & 
                                     (metadata_df["bmi"] < 30.0)]

print("Constructing IBD cohort with matched healthy samples")
ibd_cohort = buildDataSubset(healthy_population, "ibd", 1, 0)
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
antiB_cohort1 = buildDataSubset(healthy_population, "antibiotic_history", "Year", "I have not taken antibiotics in the past year.")
antiB_cohort1.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/antiB_Year_cohort.csv")

print("Constructing Antiobiotic 6 Months cohort with matched healthy samples")
antiB_cohort2 = buildDataSubset(healthy_population, "antibiotic_history", "6 months", "I have not taken antibiotics in the past year.")
antiB_cohort2.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/antiB_6Month_cohort.csv")

print("Constructing Antiobiotic Year cohort with matched healthy samples")
antiB_cohort3 = buildDataSubset(healthy_population, "antibiotic_history", "Month", "I have not taken antibiotics in the past year.")
antiB_cohort3.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/antiB_Month_cohort.csv")

print("Constructing Antiobiotic Week cohort with matched healthy samples")
antiB_cohort4 = buildDataSubset(healthy_population, "antibiotic_history", "Week", "I have not taken antibiotics in the past year.")
antiB_cohort4.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/antiB_Week_cohort.csv")


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
diabetes_cohort = buildDataSubset(healthy_population, "diabetes_type", "Type II diabetes", 0)
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

young_cohort = buildDataSubset(healthy_population_young, "age_group", 1, 0)
young_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/age_young_cohort.csv")

print("Constructing 70 and over cohort with matched healthy adults")
healthy_population_old = healthy_population[healthy_population["age_years"] >= 20.0]
healthy_population_old["age_group"] = healthy_population_old["age_years"] 
young = healthy_population_old[healthy_population_old["age_years"] <= 69.0].index
old = healthy_population_old[healthy_population_old["age_years"] >= 70.0].index
healthy_population_old.loc[young, "age_group"] = 0
healthy_population_old.loc[old, "age_group"] = 1

old_cohort = buildDataSubset(healthy_population_old, "age_group", 1, 0)
old_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/age_old_cohort.csv")

healthy_population = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                                     (metadata_df["age_years"] <= 69.0) &
                                     (metadata_df["antibiotic_history"] == "I have not taken antibiotics in the past year.") &
                                     (metadata_df["diabetes"] == 0) &
                                     (metadata_df["ibd"] == 0) &
                                     (metadata_df["bmi"] >= 18.5) & 
                                     (metadata_df["bmi"] < 30.0)]

print("Constructing Country cohort with matched healthy samples")
country_cohort = buildDataSubset(healthy_population, "country", "USA", "United Kingdom")
country_cohort.to_csv(dir_path + "Feature_Cohorts/Phase_I_Cohorts/country_cohort.csv")
print(len(country_cohort))

