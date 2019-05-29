#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:22:24 2019

@author: sklarjg
"""

#import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)

matching_features = ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]
for val in matching_features:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
metadata_df.loc[:, ["bmi", "age_years"]] = metadata_df.loc[:, ["bmi", "age_years"]].astype(float)

###keep samples from countries with sufficient number of samples 
metadata_df = metadata_df[metadata_df["country"].isin(['USA', 'United Kingdom', 'Canada'])]
metadata_df["bmi"] = metadata_df["bmi"].astype(float)
metadata_df["age_years"] = metadata_df["age_years"].astype(float)
metadata_df["weight_kg"] = metadata_df["weight_kg"].astype(float)
metadata_df["longitude"] = metadata_df["longitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)
metadata_df["latitude"] = metadata_df["latitude"].replace(["Unspecified", "Not provided"], -10.0).astype(float)
metadata_df["race"] = metadata_df["race"].replace(["Unspecified", "Not provided"], "Other")
metadata_df['sex'] = metadata_df['sex'].replace(["Unspecified", "Not provided", "unspecified"], "other")

###MATCHING METADATA FEATURES TO BUILD BALANCED COHORTS
le_race = LabelEncoder()
le_diet = LabelEncoder()
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

def buildDataSubset(metadata, target_var, pos_target, neg_target, age_related):
    ##
    ##target_var: name of metadata variable
    ##pos_target: positive class label
    ##neg_target: negavtive class label (possibly a list of labels)
    ##age_related: boolean (remove age related matching variables from matchign criteria)
    ##
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
    if age_related:
        cm = cosine_similarity(metadata_matching.loc[pos_class, :].drop(["age_years", "bmi"], axis = 1), 
                               metadata_matching.loc[neg_class, :].drop(["age_years", "bmi"], axis = 1))
    else:
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


#%%
###BMI SPECIAL FEATURE:
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 18.0) & (metadata_df["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["bmi"] <= 60.0]
print metadata_df_healthy["bmi_cat"].value_counts()

print ("Constructing Obese cohort with matched normal samples")
obese_cohort = buildDataSubset(metadata_df_healthy, "bmi_cat", "Obese", "Normal", False)
obese_cohort = metadata_matching.loc[obese_cohort, :]
obese_cohort["target"] = metadata_df_healthy.loc[obese_cohort.index, "bmi_cat"]
obese_cohort.loc[obese_cohort["target"] != "Obese", "target"] = 0
obese_cohort.loc[obese_cohort["target"] == "Obese", "target"] = 1
obese_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/Obese_cohort.csv")
print len(obese_cohort)

print ("Constructing Overweight cohort with matched normal samples")
overweight_cohort = buildDataSubset(metadata_df_healthy, "bmi_cat", "Overweight", "Normal", False)
overweight_cohort = metadata_matching.loc[overweight_cohort, :]
overweight_cohort["target"] = metadata_df_healthy.loc[overweight_cohort.index, "bmi_cat"]
overweight_cohort.loc[overweight_cohort["target"] != "Overweight", "target"] = 0
overweight_cohort.loc[overweight_cohort["target"] == "Overweight", "target"] = 1
overweight_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/Overweight_cohort.csv")
print len(overweight_cohort)

print ("Constructing Underweight cohort with matched normal samples")
underweight_cohort = buildDataSubset(metadata_df_healthy, "bmi_cat", "Underweight", "Normal", False)
underweight_cohort = metadata_matching.loc[underweight_cohort, :]
underweight_cohort["target"] = metadata_df_healthy.loc[underweight_cohort.index, "bmi_cat"]
underweight_cohort.loc[underweight_cohort["target"] != "Underweight", "target"] = 0
underweight_cohort.loc[underweight_cohort["target"] == "Underweight", "target"] = 1
underweight_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/Underweight_cohort.csv")
print len(underweight_cohort)

###IBD
metadata_df_healthy = metadata_df[(metadata_df["bmi"] < 30.0) & (metadata_df["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["age_years"] >= 18.0) & (metadata_df_healthy["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
print metadata_df_healthy["ibd"].value_counts()

print ("Constructing IBD cohort with matched healthy samples")
ibd_cohort = buildDataSubset(metadata_df_healthy, "ibd", 1, 0, False)
ibd_cohort = metadata_matching.loc[ibd_cohort, :]
ibd_cohort["target"] = metadata_df_healthy.loc[ibd_cohort.index, "ibd"]
ibd_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/IBD_cohort.csv")
print len(ibd_cohort)

##Antibiotic History
metadata_df_healthy = metadata_df[(metadata_df["bmi"] < 30.0) & (metadata_df["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["age_years"] >= 18.0) & (metadata_df_healthy["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
print metadata_df_healthy["antibiotic_history"].value_counts()

print ("Constructing Antiobiotic Year cohort with matched healthy samples")
antiB_cohort1 = buildDataSubset(metadata_df_healthy, "antibiotic_history", "Year", "I have not taken antibiotics in the past year.", False)
antiB_cohort1 = metadata_matching.loc[antiB_cohort1, :]
antiB_cohort1["target"] = metadata_df_healthy.loc[antiB_cohort1.index, "antibiotic_history"]
antiB_cohort1.loc[antiB_cohort1["target"] != "Year", "target"] = 0
antiB_cohort1.loc[antiB_cohort1["target"] == "Year", "target"] = 1
antiB_cohort1.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/antiB_Year_cohort.csv")
print len(antiB_cohort1)

print ("Constructing Antiobiotic 6 Months cohort with matched healthy samples")
antiB_cohort2 = buildDataSubset(metadata_df_healthy, "antibiotic_history", "6 months", "I have not taken antibiotics in the past year.", False)
antiB_cohort2 = metadata_matching.loc[antiB_cohort2, :]
antiB_cohort2["target"] = metadata_df_healthy.loc[antiB_cohort2.index, "antibiotic_history"]
antiB_cohort2.loc[antiB_cohort2["target"] != "6 months", "target"] = 0
antiB_cohort2.loc[antiB_cohort2["target"] == "6 months", "target"] = 1
antiB_cohort2.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/antiB_6Month_cohort.csv")
print len(antiB_cohort2)

print ("Constructing Antiobiotic Year cohort with matched healthy samples")
antiB_cohort3 = buildDataSubset(metadata_df_healthy, "antibiotic_history", "Month", "I have not taken antibiotics in the past year.", False)
antiB_cohort3 = metadata_matching.loc[antiB_cohort3, :]
antiB_cohort3["target"] = metadata_df_healthy.loc[antiB_cohort3.index, "antibiotic_history"]
antiB_cohort3.loc[antiB_cohort3["target"] != "Month", "target"] = 0
antiB_cohort3.loc[antiB_cohort3["target"] == "Month", "target"] = 1
antiB_cohort3.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/antiB_Month_cohort.csv")
print len(antiB_cohort3)

print ("Constructing Antiobiotic Week cohort with matched healthy samples")
antiB_cohort4 = buildDataSubset(metadata_df_healthy, "antibiotic_history", "Week", "I have not taken antibiotics in the past year.", False)
antiB_cohort4 = metadata_matching.loc[antiB_cohort4, :]
antiB_cohort4["target"] = metadata_df_healthy.loc[antiB_cohort4.index, "antibiotic_history"]
antiB_cohort4.loc[antiB_cohort4["target"] != "Week", "target"] = 0
antiB_cohort4.loc[antiB_cohort4["target"] == "Week", "target"] = 1
antiB_cohort4.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/antiB_Week_cohort.csv")
print len(antiB_cohort4)

##Diabetes
metadata_df_healthy = metadata_df[metadata_df["bmi"] > 12.5]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["age_years"] >= 18.0) & (metadata_df_healthy["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
print metadata_df_healthy["diabetes_type"].value_counts()

print ("Constructing Diabetes type II cohort with matched healthy samples")
diabetes_cohort = buildDataSubset(metadata_df_healthy, "diabetes_type", "Type II diabetes", ["Not provided", "Unspecified"], False)
diabetes_cohort = metadata_matching.loc[diabetes_cohort, :]
diabetes_cohort["target"] = metadata_df_healthy.loc[diabetes_cohort.index, "diabetes_type"]
diabetes_cohort.loc[diabetes_cohort["target"] != "Type II diabetes", "target"] = 0
diabetes_cohort.loc[diabetes_cohort["target"] == "Type II diabetes", "target"] = 1
diabetes_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/diabetes_typeII_cohort.csv")
print len(diabetes_cohort)

##Age
print len(metadata_df)
metadata_df_healthy = metadata_df[(metadata_df["bmi"] < 30.0) & (metadata_df["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[~metadata_df_healthy["age_years"].isin(["Not provided", "Unspecified"])]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]


print ("Constructing child [6:18] cohort with matched healthy adults")
metadata_df_healthy_young = metadata_df_healthy[metadata_df_healthy["age_years"] < 70.0]
metadata_df_healthy_young["age_group"] = metadata_df_healthy_young["age_years"] 
young = metadata_df_healthy_young[(metadata_df_healthy_young["age_years"] <= 18.0) & (metadata_df_healthy_young["age_years"] >= 6.0)].index
old = metadata_df_healthy_young[metadata_df_healthy_young["age_years"] > 18.0].index
metadata_df_healthy_young.loc[young, "age_group"] = 1
metadata_df_healthy_young.loc[old, "age_group"] = 0
young_cohort = buildDataSubset(metadata_df_healthy_young, "age_group", 1, 0, True)
young_cohort = metadata_matching.loc[young_cohort, :]
young_cohort["target"] = metadata_df_healthy_young.loc[young_cohort.index, "age_group"]
young_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/age_6-18.csv")
print len(young_cohort)


print ("Constructing toddler [3:5] cohort with matched healthy adults")
metadata_df_healthy_infant = metadata_df_healthy[metadata_df_healthy["age_years"] < 70.0]
metadata_df_healthy_infant["age_group"] = metadata_df_healthy_infant["age_years"] 
infant = metadata_df_healthy_infant[(metadata_df_healthy_infant["age_years"] <= 5.0) & (metadata_df_healthy_infant["age_years"] >= 3.0)].index
old = metadata_df_healthy_infant[metadata_df_healthy_infant["age_years"] > 18.0].index
print len(infant), len(old)
metadata_df_healthy_infant.loc[infant, "age_group"] = 1
metadata_df_healthy_infant.loc[old, "age_group"] = 0
infant_cohort = buildDataSubset(metadata_df_healthy_infant, "age_group", 1, 0, True)
infant_cohort = metadata_matching.loc[infant_cohort, :]
infant_cohort["target"] = metadata_df_healthy_infant.loc[infant_cohort.index, "age_group"]
infant_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/age_3-5.csv")
print len(infant_cohort)

print ("Constructing 2 and under cohort with matched healthy adults")
metadata_df_healthy_infant = metadata_df_healthy[metadata_df_healthy["age_years"] < 70.0]
metadata_df_healthy_infant["age_group"] = metadata_df_healthy_infant["age_years"] 
infant = metadata_df_healthy_infant[metadata_df_healthy_infant["age_years"] <= 2.0].index
old = metadata_df_healthy_infant[metadata_df_healthy_infant["age_years"] >= 18.0].index
print len(infant), len(old)
metadata_df_healthy_infant.loc[infant, "age_group"] = 1
metadata_df_healthy_infant.loc[old, "age_group"] = 0
infant_cohort = buildDataSubset(metadata_df_healthy_infant, "age_group", 1, 0, True)
infant_cohort = metadata_matching.loc[infant_cohort, :]
infant_cohort["target"] = metadata_df_healthy_infant.loc[infant_cohort.index, "age_group"]
infant_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/age_1-2.csv")
print len(infant_cohort)


print ("Constructing 65 and over cohort with matched healthy adults")
metadata_df_healthy_old = metadata_df_healthy[metadata_df_healthy["age_years"] >= 18.0]
metadata_df_healthy_old["age_group"] = metadata_df_healthy_old["age_years"] 
young = metadata_df_healthy_old[metadata_df_healthy_old["age_years"] < 65.0].index
old = metadata_df_healthy_old[metadata_df_healthy_old["age_years"] >= 65.0].index
metadata_df_healthy_old.loc[young, "age_group"] = 0
metadata_df_healthy_old.loc[old, "age_group"] = 1
old_cohort = buildDataSubset(metadata_df_healthy_old, "age_group", 1, 0, True)
old_cohort = metadata_matching.loc[old_cohort, :]
old_cohort["target"] = metadata_df_healthy_old.loc[old_cohort.index, "age_group"]
old_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/age_over65.csv")
print len(old_cohort)

metadata_df_healthy = metadata_df_healthy.loc[:, ["age_years", "age_cat", "sex", "bmi", "longitude", "latitude", "race"]]
metadata_df_healthy.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/age_healthy_cohort.csv")


#%%
## sliding age window test:
import numpy as np
metadata_df_healthy = metadata_df[(metadata_df["bmi"] < 30.0) & (metadata_df["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[~metadata_df_healthy["age_years"].isin(["Not provided", "Unspecified"])]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["age_years"] <= 70.0]

for n,i in enumerate(range(4, 41)):
    age_window = range(i-3, i+1)    
    age_window = [float(age) for age in age_window]
    control_min = 41
    print n, age_window, control_min
    young = metadata_df_healthy[metadata_df_healthy["age_years"].isin(age_window)].index.values
    old = metadata_df_healthy[metadata_df_healthy["age_years"] >= control_min].index.values
    
    sample_population = metadata_df_healthy.loc[np.concatenate([old, young]), :]
    sample_population["age_group"] = sample_population["age_years"] 
    sample_population.loc[young, "age_group"] = 1
    sample_population.loc[old, "age_group"] = 0

    young_cohort = buildDataSubset(sample_population, "age_group", 1, 0, True)
    young_cohort = metadata_matching.loc[young_cohort, :]
    young_cohort["target"] = sample_population.loc[young_cohort.index, "age_group"]
    young_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/age_cohorts/age_under" + str(i+1) + ".csv")

#%%

##Gender
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 18.0) & (metadata_df["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["bmi"] < 30.0) & (metadata_df_healthy["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["sex"] != "other"]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]

gender_healthy_cohort = buildDataSubset(metadata_df_healthy, "sex", "male", "female", False)
gender_healthy_cohort = metadata_matching.loc[gender_healthy_cohort, :]
gender_healthy_cohort["target"] = metadata_df_healthy.loc[gender_healthy_cohort.index, "sex"]
gender_healthy_cohort.loc[gender_healthy_cohort["target"] == "female", "target"] = 0
gender_healthy_cohort.loc[gender_healthy_cohort["target"] == "male", "target"] = 1
gender_healthy_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/gender_healthy_cohort.csv")


###Alcohol_Consumption Frequeny:
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 18.0) & (metadata_df["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["bmi"] < 30.0) & (metadata_df_healthy["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
print metadata_df_healthy["alcohol_frequency"].value_counts()


print ("Constructing alcohol rare cohort with matched healthy samples")
alc_cohort = buildDataSubset(metadata_df_healthy, "alcohol_frequency", 1, 0, False)
alc_cohort = metadata_matching.loc[alc_cohort, :]
alc_cohort["target"] = metadata_df_healthy.loc[alc_cohort.index, "alcohol_frequency"]
alc_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/alcohol_rare_cohort.csv")


print ("Constructing alcohol occasional cohort with matched healthy samples")
alc_cohort = buildDataSubset(metadata_df_healthy, "alcohol_frequency", 2, 0, False)
alc_cohort = metadata_matching.loc[alc_cohort, :]
alc_cohort["target"] = metadata_df_healthy.loc[alc_cohort.index, "alcohol_frequency"]
alc_cohort.loc[alc_cohort["target"] == 2, "target"] = 1
alc_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/alcohol_occasional_cohort.csv")


print ("Constructing alcohol regular cohort with matched healthy samples")
alc_cohort = buildDataSubset(metadata_df_healthy, "alcohol_frequency", 3, 0, False)
alc_cohort = metadata_matching.loc[alc_cohort, :]
alc_cohort["target"] = metadata_df_healthy.loc[alc_cohort.index, "alcohol_frequency"]
alc_cohort.loc[alc_cohort["target"] == 3, "target"] = 1
alc_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/alcohol_regular_cohort.csv")


print ("Constructing alcohol daily with matched healthy samples")
alc_cohort = buildDataSubset(metadata_df_healthy, "alcohol_frequency", 4, 0, False)
alc_cohort = metadata_matching.loc[alc_cohort, :]
alc_cohort["target"] = metadata_df_healthy.loc[alc_cohort.index, "alcohol_frequency"]
alc_cohort.loc[alc_cohort["target"] == 4, "target"] = 1
alc_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/alcohol_daily_cohort.csv")

#%%

###Contraceptive (just women)
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 18.0) & (metadata_df["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["bmi"] < 30.0) & (metadata_df_healthy["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["sex"] == "female"]
print metadata_df_healthy["contraceptive"].value_counts()

print ("Constructing contraceptive pill cohort with matched healthy samples")
pill_cohort = buildDataSubset(metadata_df_healthy, "contraceptive", "Yes, I am taking the \"pill\"", "No", False)
pill_cohort = metadata_matching.loc[pill_cohort, :]
pill_cohort["target"] = metadata_df_healthy.loc[pill_cohort.index, "contraceptive"]
pill_cohort.loc[pill_cohort["target"] != "Yes, I am taking the \"pill\"", "target"] = 0
pill_cohort.loc[pill_cohort["target"] == "Yes, I am taking the \"pill\"", "target"] = 1
pill_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/pill_cohort.csv")


print ("Constructing IUD cohort with matched healthy samples")
iud_cohort = buildDataSubset(metadata_df_healthy, "contraceptive", "Yes, I use a hormonal IUD (Mirena)", "No", False)
iud_cohort = metadata_matching.loc[iud_cohort, :]
iud_cohort["target"] = metadata_df_healthy.loc[iud_cohort.index, "contraceptive"]
iud_cohort.loc[iud_cohort["target"] != "Yes, I use a hormonal IUD (Mirena)", "target"] = 0
iud_cohort.loc[iud_cohort["target"] == "Yes, I use a hormonal IUD (Mirena)", "target"] = 1
iud_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/iud_cohort.csv")


####Autism: (just children)
metadata_df_healthy = metadata_df[metadata_df["age_years"] <= 30.0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["bmi"] < 30.0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]

print ("Constructing Autism cohort with matched healthy samples (under 30)")
asd_cohort = buildDataSubset(metadata_df_healthy, "asd", 1, 0, False)
asd_cohort = metadata_matching.loc[asd_cohort, :]
asd_cohort["target"] = metadata_df_healthy.loc[asd_cohort.index, "asd"]
asd_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/asd_cohort.csv")

####Gluten related groups 
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 18.0) & (metadata_df["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["bmi"] < 30.0) & (metadata_df_healthy["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
print metadata_df_healthy["gluten"].value_counts()


print ("Constructing no gluten cohort with matched healthy samples")
no_gluten_cohort = buildDataSubset(metadata_df_healthy, "gluten", "I do not eat gluten because it makes me feel bad", "No", False)
no_gluten_cohort = metadata_matching.loc[no_gluten_cohort, :]
no_gluten_cohort["target"] = metadata_df_healthy.loc[no_gluten_cohort.index, "gluten"]
no_gluten_cohort.loc[no_gluten_cohort["target"] != "I do not eat gluten because it makes me feel bad", "target"] = 0
no_gluten_cohort.loc[no_gluten_cohort["target"] == "I do not eat gluten because it makes me feel bad", "target"] = 1
no_gluten_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/no_gluten_cohort.csv")

print ("Constructing celiac cohort with matched healthy samples")
celiac_cohort = buildDataSubset(metadata_df_healthy, "gluten", "I was diagnosed with celiac disease", "No", False)
celiac_cohort = metadata_matching.loc[celiac_cohort, :]
celiac_cohort["target"] = metadata_df_healthy.loc[celiac_cohort.index, "gluten"]
celiac_cohort.loc[celiac_cohort["target"] != "I was diagnosed with celiac disease", "target"] = 0
celiac_cohort.loc[celiac_cohort["target"] == "I was diagnosed with celiac disease", "target"] = 1
celiac_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/celiac_cohort.csv")

print ("Constructing gluten_allergy cohort with matched healthy samples")
gluten_alergy_cohort = buildDataSubset(metadata_df_healthy, "gluten", "I was diagnosed with gluten allergy (anti-gluten IgG), but not celiac disease", "No", False)
gluten_alergy_cohort = metadata_matching.loc[gluten_alergy_cohort, :]
gluten_alergy_cohort["target"] = metadata_df_healthy.loc[gluten_alergy_cohort.index, "gluten"]
gluten_alergy_cohort.loc[gluten_alergy_cohort["target"] != "I was diagnosed with gluten allergy (anti-gluten IgG), but not celiac disease", "target"] = 0
gluten_alergy_cohort.loc[gluten_alergy_cohort["target"] == "I was diagnosed with gluten allergy (anti-gluten IgG), but not celiac disease", "target"] = 1
gluten_alergy_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/gluten_alergy_cohort.csv")


###USA IMMIGRANTS 
print ("Constructing USA immigrant cohort with matched healthy samples")
metadata_df_healthy["immigrant"] = metadata_df_healthy["country_of_birth"] 
immigrants_us = metadata_df_healthy[(metadata_df_healthy["country_of_birth"] != "United States") & (metadata_df_healthy["country_residence"] == "United States")].index
native_us = metadata_df_healthy[(metadata_df_healthy["country_of_birth"] == "United States") & (metadata_df_healthy["country_residence"] == "United States")].index
metadata_df_healthy.loc[immigrants_us, "immigrant"] = 1
metadata_df_healthy.loc[native_us, "immigrant"] = 0
usa_immigrant_cohort = buildDataSubset(metadata_df_healthy, "immigrant", 1, 0, False)
usa_immigrant_cohort = metadata_matching.loc[usa_immigrant_cohort, :]
usa_immigrant_cohort["target"] = metadata_df_healthy.loc[usa_immigrant_cohort.index, "immigrant"]
usa_immigrant_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/usa_immigrant_cohort.csv")


###UK IMMIGRANTS 
print ("Constructing UK immigrant cohort with matched healthy samples")
metadata_df_healthy["immigrant"] = metadata_df_healthy["country_of_birth"] 
immigrants_uk = metadata_df_healthy[(metadata_df_healthy["country_of_birth"] != "United Kingdom") & (metadata_df_healthy["country_residence"] == "United Kingdom")].index
native_uk = metadata_df_healthy[(metadata_df_healthy["country_of_birth"] == "United Kingdom") & (metadata_df_healthy["country_residence"] == "United Kingdom")].index
metadata_df_healthy.loc[immigrants_uk, "immigrant"] = 1
metadata_df_healthy.loc[native_uk, "immigrant"] = 0
uk_immigrant_cohort = buildDataSubset(metadata_df_healthy, "immigrant", 1, 0, False)
uk_immigrant_cohort = metadata_matching.loc[uk_immigrant_cohort, :]
uk_immigrant_cohort["target"] = metadata_df_healthy.loc[uk_immigrant_cohort.index, "immigrant"]
uk_immigrant_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/uk_immigrant_cohort.csv")

###DIET TYPE GROUPS
print ("Constructing diet:  cohort with matched healthy samples")
omnivore_noRed_cohort = buildDataSubset(metadata_df_healthy, "diet_type", "Omnivore but do not eat red meat", "Omnivore", False)
omnivore_noRed_cohort = metadata_matching.loc[omnivore_noRed_cohort, :]
omnivore_noRed_cohort["target"] = metadata_df_healthy.loc[omnivore_noRed_cohort.index, "diet_type"]
omnivore_noRed_cohort.loc[omnivore_noRed_cohort["target"] != "Omnivore but do not eat red meat", "target"] = 0
omnivore_noRed_cohort.loc[omnivore_noRed_cohort["target"] == "Omnivore but do not eat red meat", "target"] = 1
omnivore_noRed_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/omnivore_noRed_cohort.csv")



print ("Constructing diet: Pescatarian cohort with matched healthy samples")
pescatarian_cohort = buildDataSubset(metadata_df_healthy, "diet_type", "Vegetarian but eat seafood", "Omnivore", False)
pescatarian_cohort = metadata_matching.loc[pescatarian_cohort, :]
pescatarian_cohort["target"] = metadata_df_healthy.loc[pescatarian_cohort.index, "diet_type"]
pescatarian_cohort.loc[pescatarian_cohort["target"] != "Vegetarian but eat seafood", "target"] = 0
pescatarian_cohort.loc[pescatarian_cohort["target"] == "Vegetarian but eat seafood", "target"] = 1
pescatarian_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/pescatarian_cohort.csv")


print ("Constructing diet: Vegetarian cohort with matched healthy samples")
vegetarian_cohort = buildDataSubset(metadata_df_healthy, "diet_type", "Vegetarian", "Omnivore", False)
vegetarian_cohort = metadata_matching.loc[vegetarian_cohort, :]
vegetarian_cohort["target"] = metadata_df_healthy.loc[vegetarian_cohort.index, "diet_type"]
vegetarian_cohort.loc[vegetarian_cohort["target"] != "Vegetarian", "target"] = 0
vegetarian_cohort.loc[vegetarian_cohort["target"] == "Vegetarian", "target"] = 1
vegetarian_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/vegetarian_cohort.csv")


print ("Constructing diet: Vegan cohort with matched healthy samples")
vegan_cohort = buildDataSubset(metadata_df_healthy, "diet_type", "Vegan", "Omnivore", False)
vegan_cohort = metadata_matching.loc[vegan_cohort, :]
vegan_cohort["target"] = metadata_df_healthy.loc[vegan_cohort.index, "diet_type"]
vegan_cohort.loc[vegan_cohort["target"] != "Vegan", "target"] = 0
vegan_cohort.loc[vegan_cohort["target"] == "Vegan", "target"] = 1
vegan_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/vegan_cohort.csv")

print ("Constructing BMF 1 cohort with matched healthy samples")
bmf_less1_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Less than one", "One", False)
bmf_less1_cohort = metadata_matching.loc[bmf_less1_cohort, :]
bmf_less1_cohort["target"] = metadata_df_healthy.loc[bmf_less1_cohort.index, "bowel_movement_frequency"]
bmf_less1_cohort.loc[bmf_less1_cohort["target"] != "Less than one", "target"] = 0
bmf_less1_cohort.loc[bmf_less1_cohort["target"] == "Less than one", "target"] = 1
bmf_less1_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/bmf_less1_cohort.csv")


print ("Constructing BMF 2 cohort with matched healthy samples")
bmf_2_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Two", "One", False)
bmf_2_cohort = metadata_matching.loc[bmf_2_cohort, :]
bmf_2_cohort["target"] = metadata_df_healthy.loc[bmf_2_cohort.index, "bowel_movement_frequency"]
bmf_2_cohort.loc[bmf_2_cohort["target"] != "Two", "target"] = 0
bmf_2_cohort.loc[bmf_2_cohort["target"] == "Two", "target"] = 1
bmf_2_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/bmf_2_cohort.csv")


print ("Constructing BMF 3 cohort with matched healthy samples")
bmf_3_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Three", "One", False)
bmf_3_cohort = metadata_matching.loc[bmf_3_cohort, :]
bmf_3_cohort["target"] = metadata_df_healthy.loc[bmf_3_cohort.index, "bowel_movement_frequency"]
bmf_3_cohort.loc[bmf_3_cohort["target"] != "Three", "target"] = 0
bmf_3_cohort.loc[bmf_3_cohort["target"] == "Three", "target"] = 1
bmf_3_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/bmf_3_cohort.csv")

print ("Constructing BMF 4 cohort with matched healthy samples")
bmf_4_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Four", "One", False)
bmf_4_cohort = metadata_matching.loc[bmf_4_cohort, :]
bmf_4_cohort["target"] = metadata_df_healthy.loc[bmf_4_cohort.index, "bowel_movement_frequency"]
bmf_4_cohort.loc[bmf_4_cohort["target"] != "Four", "target"] = 0
bmf_4_cohort.loc[bmf_4_cohort["target"] == "Four", "target"] = 1
bmf_4_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/bmf_4_cohort.csv")

print ("Constructing BMF 5 cohort with matched healthy samples")
bmf_5_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_frequency", "Five or more", "One", False)
bmf_5_cohort = metadata_matching.loc[bmf_5_cohort, :]
bmf_5_cohort["target"] = metadata_df_healthy.loc[bmf_5_cohort.index, "bowel_movement_frequency"]
bmf_5_cohort.loc[bmf_5_cohort["target"] != "Five or more", "target"] = 0
bmf_5_cohort.loc[bmf_5_cohort["target"] == "Five or more", "target"] = 1
bmf_5_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/bmf_5_cohort.csv")

print ("Constructing BMQ Solid cohort with matched healthy samples")
bmq_solid_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_quality",
                                   ["I tend to be constipated (have difficulty passing stool) - Type 1 and 2", "I tend to be constipated (have difficulty passing stool)"],
                                   ["I tend to have normal formed stool - Type 3 and 4" , "I tend to have normal formed stool"], False)
bmq_solid_cohort = metadata_matching.loc[bmq_solid_cohort, :]
bmq_solid_cohort["target"] = metadata_df_healthy.loc[bmq_solid_cohort.index, "bowel_movement_quality"]
bmq_solid_cohort.loc[~bmq_solid_cohort["target"].isin(["I tend to be constipated (have difficulty passing stool) - Type 1 and 2", "I tend to be constipated (have difficulty passing stool)"]), "target"] = 0
bmq_solid_cohort.loc[bmq_solid_cohort["target"].isin(["I tend to be constipated (have difficulty passing stool) - Type 1 and 2", "I tend to be constipated (have difficulty passing stool)"]), "target"] = 1
bmq_solid_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/bmq_solid_cohort.csv")

print ("Constructing BMQ loose cohort with matched healthy samples")
bmq_loose_cohort = buildDataSubset(metadata_df_healthy, "bowel_movement_quality", 
                                   ["I tend to have diarrhea (watery stool) - Type 5, 6 and 7" , "I tend to have diarrhea (watery stool)"], 
                                   ["I tend to have normal formed stool - Type 3 and 4" , "I tend to have normal formed stool"], False)
bmq_loose_cohort = metadata_matching.loc[bmq_loose_cohort, :]
bmq_loose_cohort["target"] = metadata_df_healthy.loc[bmq_loose_cohort.index, "bowel_movement_quality"]
bmq_loose_cohort.loc[~bmq_loose_cohort["target"].isin(["I tend to have diarrhea (watery stool) - Type 5, 6 and 7" , "I tend to have diarrhea (watery stool)"]), "target"] = 0
bmq_loose_cohort.loc[bmq_loose_cohort["target"].isin(["I tend to have diarrhea (watery stool) - Type 5, 6 and 7" , "I tend to have diarrhea (watery stool)"]), "target"] = 1
bmq_loose_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/bmq_loose_cohort.csv")


print ("Constructing weight increase cohort with matched healthy samples")
weigth_increase_cohort = buildDataSubset(metadata_df_healthy, "weight_change", "Increased more than 10 pounds", "Remained stable", False)
weigth_increase_cohort = metadata_matching.loc[weigth_increase_cohort, :]
weigth_increase_cohort["target"] = metadata_df_healthy.loc[weigth_increase_cohort.index, "weight_change"]
weigth_increase_cohort.loc[weigth_increase_cohort["target"] != "Increased more than 10 pounds", "target"] = 0
weigth_increase_cohort.loc[weigth_increase_cohort["target"] == "Increased more than 10 pounds", "target"] = 1
weigth_increase_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/weigth_increase_cohort.csv")

print ("Constructing weight decrease cohort with matched healthy samples")
weigth_decrease_cohort = buildDataSubset(metadata_df_healthy, "weight_change", "Decreased more than 10 pounds", "Remained stable", False)
weigth_decrease_cohort = metadata_matching.loc[weigth_decrease_cohort, :]
weigth_decrease_cohort["target"] = metadata_df_healthy.loc[weigth_decrease_cohort.index, "weight_change"]
weigth_decrease_cohort.loc[weigth_decrease_cohort["target"] != "Decreased more than 10 pounds", "target"] = 0
weigth_decrease_cohort.loc[weigth_decrease_cohort["target"] == "Decreased more than 10 pounds", "target"] = 1
weigth_decrease_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/weigth_decrease_cohort.csv")

#"collection_season"




#%%
#####RACE
metadata_df_healthy = metadata_df[(metadata_df["age_years"] >= 18.0) & (metadata_df["age_years"] <= 80.0)]
metadata_df_healthy = metadata_df_healthy[(metadata_df_healthy["bmi"] < 30.0) & (metadata_df_healthy["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]
print metadata_df_healthy["race"].value_counts()

#["Caucasian", "Asian or Pacific Islander", "Hispanic", "African American"]

print ("Constructing caucasian cohort with matched healthy samples")
cauc_cohort = buildDataSubset(metadata_df_healthy, "race", "Caucasian", ["Asian or Pacific Islander", "Hispanic", "African American"], False)
cauc_cohort = metadata_matching.loc[cauc_cohort, :]
cauc_cohort["target"] = metadata_df_healthy.loc[cauc_cohort.index, "race"]
cauc_cohort.loc[cauc_cohort["target"] != "Caucasian", "target"] = "Control"
cauc_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/caucasian_cohort.csv")

print ("Constructing asian cohort with matched healthy samples")
asian_cohort = buildDataSubset(metadata_df_healthy, "race", "Asian or Pacific Islander", ["Caucasian", "Hispanic", "African American"], False)
asian_cohort = metadata_matching.loc[asian_cohort, :]
asian_cohort["target"] = metadata_df_healthy.loc[asian_cohort.index, "race"]
asian_cohort.loc[asian_cohort["target"] != "Asian or Pacific Islander", "target"] = "Control"
asian_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/asian_cohort.csv")

print ("Constructing AA cohort with matched healthy samples")
africanA_cohort = buildDataSubset(metadata_df_healthy, "race", "African American", ["Caucasian", "Asian or Pacific Islander", "Hispanic"], False)
africanA_cohort = metadata_matching.loc[africanA_cohort, :]
africanA_cohort["target"] = metadata_df_healthy.loc[africanA_cohort.index, "race"]
africanA_cohort.loc[africanA_cohort["target"] != "African American", "target"] = "Control"
africanA_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/african_american_cohort.csv")

print ("Constructing hispanic cohort with matched healthy samples")
hispanic_cohort = buildDataSubset(metadata_df_healthy, "race", "Hispanic", ["Caucasian", "Asian or Pacific Islander", "African American"], False)
hispanic_cohort = metadata_matching.loc[hispanic_cohort, :]
hispanic_cohort["target"] = metadata_df_healthy.loc[hispanic_cohort.index, "race"]
hispanic_cohort.loc[hispanic_cohort["target"] != "Hispanic", "target"] = "Control"
hispanic_cohort.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_cohorts/hispanic_cohort.csv")




