#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:34:15 2019

@author: sklarjg
"""

#IMPORTS
import pandas as pd 
import numpy as np


###FUNCTIONS USED TO CLEAN EACH MAIN GROUP OF FEAUTURE TYPES: (BINARY, CLINICAL DISEASE, FREQUENCY, AND OTHER)
def clean_disease_feature(val):
    val = str(val)
    if val.lower() == 'i do not have this condition':
        val = 0
    elif val.lower() == 'diagnosed by a medical professional (doctor, physician assistant)':
        val = 1
    elif val.lower() == 'diagnosed by an alternative medicine practitioner':
        val = 2
    elif val.lower() == 'self-diagnosed':
        val = 3
    elif val.lower() == "unspecified" or val.lower() == "not provided" or val.lower() == "nan":
        val = 4
    else:
        val = val
    return val

def clean_frequency_feature(val):
    val = str(val)
    if val.lower() == 'never':
        val = 0
    elif val.lower() == 'rarely (a few times/month)' or val.lower() == 'rarely (less than once/week)':
        val = 1
    elif val.lower() == 'occasionally (1-2 times/week)':
        val = 2
    elif val.lower() == 'regularly (3-5 times/week)':
        val = 3
    elif val.lower() == 'daily':
        val = 4
    elif val.lower() == "unspecified" or val.lower() == "not provided" or val.lower() == "nan":
        val = 5
    else:
        val = val
    return val

def clean_binary_feature(val):
    val = str(val)
    if val.lower() == 'true' or val.lower() == 'yes':
        val = 1
    elif val.lower() == 'false' or val.lower() == 'no':
        val = 0
    elif val.lower() == "unspecified" or val.lower() == "not provided" or val.lower() == 'not sure' or val.lower() == 'nan':
        val = 2
    else:
        val = 2
    return val

###LOAD OTU DATA, HOST METADATA, AND SAMPLE METADATA:
metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/AGP_Metadata.csv", index_col = 1, low_memory=False)
otu_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/AGP_merged2_s50_rar10k_forR_md5.txt", sep = "\t", index_col = 0)
otu_df = otu_df.T
otu_df.index = otu_df.index.str.split(".").str[0]
file_data_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/AGP_File_Metadata.csv")
file_data_df["LibraryName"] = file_data_df["LibraryName"].str.replace(".", "_")
file_data_df.index = file_data_df["Run"]
metadata_df["AGP_ID"] = metadata_df.index

###KEEP FILE DATA FOR FILES THAT MADE IT THROUGH PROCESSING SCRIPTS
file_data_df = file_data_df[file_data_df.index.isin(otu_df.index)]

###REMOVE EXTRA SAMPLES/METADATA FROM INDIVIDUALS THAT WERE SEQUENCED MORE THAN ONCE
file_dups = file_data_df[file_data_df["LibraryName"].duplicated(keep=False)].sort_values(by = "LibraryName")
file_data_df = file_data_df[~file_data_df["LibraryName"].duplicated(keep='first')].sort_values(by = "LibraryName")
otu_df = otu_df[otu_df.index.isin(file_data_df.index)]
metadata_df = metadata_df[metadata_df.index.isin(file_data_df["LibraryName"])]
sample_id_map = pd.Series(file_data_df["Run"].values, index = file_data_df["LibraryName"]).to_dict()
metadata_df.index = metadata_df.index.map(sample_id_map)

###Get feature summaries for deciding which features to classify from otu data
data_summary = metadata_df.describe(include='all')
data_summary = data_summary.T
data_summary = data_summary[data_summary["unique"] > 1]
data_features = data_summary.index

###Get subset of samples that 
vioscreen_features = [val for val in data_features if 'vioscreen' in val]
fermented_features =  [val for val in data_features if 'fermented' in val]#
fermented_features.remove("fermented_plant_frequency")


vioscreen_data_df = metadata_df.loc[:,metadata_df.columns.isin(vioscreen_features)]
vioscreen_data_df = vioscreen_data_df[~vioscreen_data_df["vioscreen_a_bev"].isin(["Not provided", np.nan, "Unspecified"])]

##REMOVE VIOSCREEN AND FERMENTED FEATURES FROM THE STANDARD METADATA DATAFRAME (NOT ENOUGH PEOPLE OPTED TO FILL IT OUT)
data_summary = data_summary[~data_summary.index.isin(vioscreen_features)]
data_summary = data_summary[~data_summary.index.isin(fermented_features)]

###REMOVE VIOSCREEN FEATURES THAT ARE MOSTLY EMPTY
num_rows = len(vioscreen_data_df)
print num_rows
print 
for col in vioscreen_data_df.columns:
    counts = vioscreen_data_df[col].value_counts()
    num_nan = vioscreen_data_df[col].isnull().sum(axis = 0)
    if num_nan > num_rows/2:
        vioscreen_data_df.drop(col, axis = 1, inplace = True)
    if "0.0" in counts.index:
        if  counts["0.0"]  >= num_rows/2:
            print col, counts["0.0"]
            vioscreen_data_df.drop(col, axis = 1, inplace = True)
    if "Not provided" in counts.index:     
        if  counts["Not provided"]  > num_rows/2:
            vioscreen_data_df.drop(col, axis = 1, inplace = True)

vio_samples = vioscreen_data_df.index
metadata_vioscreen_df = metadata_df[metadata_df.index.isin(vio_samples)] 

meta_keep = data_summary.index
metadata_df = metadata_df[meta_keep]

###GROUP FEATURES BY COMMON ANSWERS
disease_features = []
binary_features = []
frequency_features = []
other_features = []

for val in metadata_df.columns:
    col_counts = metadata_df[val].value_counts()
    if col_counts.index.contains("Diagnosed by a medical professional (doctor, physician assistant)"):
        disease_features.append(val)
    elif col_counts.index.contains("Occasionally (1-2 times/week)"):
        frequency_features.append(val)
    elif col_counts.index.contains("Yes"):
        binary_features.append(val)
    else:
        other_features.append(val)


print "disease_features", len(disease_features)
print disease_features
print 
print "binary_feautures", len(binary_features)
print binary_features
print 
print "frequency_features", len(frequency_features)
print frequency_features
print 
print "other_features", len(other_features)
print other_features

####INTEGER ENCODE ALL FEATURES THAT FIT INTO ONE OF THE THREE MAIN GROUPS
####OTHER FEATURES WILL BE ENCODED SEPERATELY 
for val in disease_features:
    metadata_df.loc[:, val] = metadata_df.loc[:, val].apply(clean_disease_feature)
for val in frequency_features:
    metadata_df.loc[:, val] = metadata_df.loc[:, val].apply(clean_frequency_feature)
for val in binary_features:
    metadata_df.loc[:, val] = metadata_df.loc[:, val].apply(clean_binary_feature)

####SUBSET OF OTHER FEATURES DETERMINED TO BE INTERESTING, REST ARE REMOVED
other_features = ['age_cat', 'antibiotic_history', 'bmi_cat', 'bowel_movement_frequency', 
                  'bowel_movement_quality', 'cancer_treatment', 'census_region', 'collection_season', 'contraceptive', 
                  "flu_vaccine_date", 'gluten', 'country', 'diabetes_type', 'drinks_per_session', 'drinking_water_source', "types_of_plants",
                  'diet_type', 'elevation', 'fed_as_infant', 'level_of_education', 'race', 'sex','sleep_duration', 
                  'state', 'weight_change', 'country_of_birth', 'country_residence', 'subset_healthy', 'survey_id']
regression_features = ['age_years', 'age_corrected', 'bmi', 'bmi_corrected', 'weight_kg', 'longitude', 'latitude']
meta_keep = np.concatenate((binary_features,frequency_features,disease_features, other_features, regression_features))
metadata_df = metadata_df[meta_keep]

#####FEATURES THAT SHOULD NOT BE CLASSIFIED EITHER BECAUSE OF REDUNDANCEY OR BECAUSE THEY SPECIFY THE LACK OF INFORMATION 
unspecified_features = ['alcohol_types_unspecified', 'allergic_to_unspecified', 'mental_illness_type_unspecified', 
                        'non_food_allergies_unspecified', 'specialized_diet_unspecified', 'specialized_diet_i_do_not_eat_a_specialized_diet',
                        'dna_extracted','physical_specimen_remaining','public', 'assigned_from_geo']
metadata_df.drop(unspecified_features, axis = 1, inplace = True)

#%%

##UPDATE AFTER REMOVING ADDITIONAL FEATURES
meta_keep = metadata_df.columns.values
binary_features = [val for val in binary_features if val in meta_keep]

feature_groups = pd.DataFrame([], index = meta_keep, columns = ["group"])
feature_groups.loc[binary_features, "group"] = "binary"
feature_groups.loc[frequency_features, "group"] = "frequency"
feature_groups.loc[disease_features, "group"] = "disease"
feature_groups.loc[other_features, "group"] = "other"
feature_groups.loc[regression_features, "group"] = "regression"

metadata_temporal_df = metadata_df[metadata_df.duplicated(subset= "survey_id", keep= False)]
metadata_temporal_df = metadata_temporal_df[meta_keep]
metadata_df = metadata_df.drop_duplicates(subset='survey_id', keep="last")
metadata_vioscreen_df = metadata_vioscreen_df.drop_duplicates(subset='survey_id', keep="last")
metadata_vioscreen_df = metadata_vioscreen_df[meta_keep]

otu_temporal_df = otu_df[otu_df.index.isin(metadata_temporal_df.index)]
otu_vioscreen_df = otu_df[otu_df.index.isin(metadata_vioscreen_df.index)]
otu_unique_df = otu_df[otu_df.index.isin(metadata_df.index)]

##Save Cleaned Data!
#feature_groups.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_groups.csv")

metadata_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv")
#metadata_vioscreen_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/VioScreen_Metadata.csv")
#metadata_temporal_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Temporal_Metadata.csv")
#vioscreen_data_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/VioScreen_Data.csv")

#otu_temporal_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Temporal_Otu_Data.csv")
#otu_vioscreen_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_VioScreen_Otu_Data.csv")
#otu_unique_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Otu_Data.csv")
