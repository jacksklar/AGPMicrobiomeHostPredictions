#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:37:13 2019

@author: sklarjg
"""

import os
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns        


metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
file_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/AGP_File_Metadata.csv", index_col = 0)
otufull_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/AGP_merged2_s50_rar10k_forR_md5.txt", sep = "\t").T

print "Raw Fastq reads downloaded from ENA:", len(file_df)
print "Participants (after rarification):", len(otufull_df)
print "Participants (no duplicates, only gut samples):", len(metadata_df)

matching_features = ["diabetes", "age_years", "bmi", "ibd", "antibiotic_history"]
for val in matching_features:
    metadata_df = metadata_df[~metadata_df[val].isin(["Not provided","Unspecified", 4, 3, 2])]
metadata_df.loc[:, ["bmi", "age_years"]] = metadata_df.loc[:, ["bmi", "age_years"]].astype(float)
print "Metadata (removed samples with unspecified matching metadata [diabetes, age_years, bmi, ibd, antibiotic_history]):", len(metadata_df)

###keep samples from countries with sufficient number of samples 
metadata_df = metadata_df[metadata_df["country"].isin(['USA', 'United Kingdom', 'Canada'])]
print "Metadata (samples from Countries: USA, UK, AUS, CAN):", len(metadata_df)

metadata_df_healthy = metadata_df[(metadata_df["bmi"] < 30.0) & (metadata_df["bmi"] > 12.5)]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["ibd"] == 0]
metadata_df_healthy = metadata_df_healthy[~metadata_df_healthy["age_years"].isin(["Not provided", "Unspecified"])]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["diabetes"] == 0]
metadata_df_healthy = metadata_df_healthy[metadata_df_healthy["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])]


print "Metadata Healthy Control Poputlation:", len(metadata_df_healthy)
#%%

no_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 0]["age_years"]
rare_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 1]["age_years"]
occ_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 2]["age_years"]
reg_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 3]["age_years"]
day_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 4]["age_years"]
full_bins = len(metadata_df_healthy["age_years"].unique()) + 1
full_bins = 96
plt.figure(figsize = (10,6))

plt.hist([day_alc, reg_alc, occ_alc, rare_alc, no_alc],
         color = ["blue", "green", "yellow", "orange", "red"],
         label = ["Daily", "Regularly", "Occasionally", "Rarely", "Never"],
         bins = full_bins, stacked=True, density=True)
plt.legend()
plt.title("Alcohol consumption frequency in healthy population")
plt.xlabel("Age")
plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/alcohol_freq_hist.pdf")
plt.show()

#%%

bin_res1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results/rf_results.csv", index_col = 0)
bin_res2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_updated/rf_results.csv", index_col = 0)

bin_auc1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results/AUCs/rf_aucs.csv", index_col = 0)
bin_auc2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_updated/AUCs/rf_aucs.csv", index_col = 0)
feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)


standard_match_vars = ["alcohol_rare_cohort","Underweight_cohort",
                       "pescatarian_cohort","diabetes_typeII_cohort","ibs","Overweight_cohort",
                       'alcohol_occasional_cohort','antiB_6Month_cohort', u'antiB_Month_cohort',
                       u'antiB_Week_cohort', u'antiB_Year_cohort', "age_over65" ]

temp = bin_res1[((bin_res1["p_val"] <= 0.05) & (bin_res1["auc_median"] >= 0.65)) | bin_res1.index.isin(standard_match_vars)].sort_values("auc_median", ascending = False)
boxplotdata1 = bin_auc1.loc[temp.index, :].values
boxplotdata1 = pd.DataFrame(boxplotdata1, index = feature_info.loc[temp.index, "plot_name"]).T
boxplotdata2 = bin_auc2.loc[temp.index, :].values
boxplotdata2 = pd.DataFrame(boxplotdata2).T 

print "Standard Matching:"
print
pval1 = bin_res1.loc[temp.index,"p_val"]
print "Not Significant:"
print pval1[pval1 > 0.05]
print
print
print "New Matching:"
print
pval2 = bin_res2.loc[temp.index,"p_val"]
print "Not Significant:"
print pval2[pval2 > 0.05]

f, (ax1, ax2) = plt.subplots(1, 2)                                            
f.set_figheight(8)
f.set_figwidth(13)
sns.boxplot(data = boxplotdata1, notch = False, showfliers=False, palette = "Blues_r", orient = "h", ax = ax1)
sns.set()
ax1.set_xlim(0.5, 1.0)
ax1.set_ylabel('')
ax1.set_xlabel('AUC')
ax1.set_title("Standard Matching")

g = sns.boxplot(data = boxplotdata2, notch = False, showfliers=False, palette = "Blues_r", orient = "h", ax = ax2)
g.set_yticks([])
ax2.set_xlim(0.5, 1.0)
ax2.set_title("New Matching")
ax2.set_xlabel('AUC')

plt.subplots_adjust(wspace=0.1)
#plt.tight_layout()
plt.gcf().subplots_adjust(left=0.4)

plt.savefig("/Users/sklarjg/Desktop/new_rf_results.pdf", format='pdf')
plt.show()

#%%
freq_res1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results/rf_results.csv", index_col = 0)
freq_res2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_updated/rf_results.csv", index_col = 0)

freq_auc1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results/AUCs/rf_aucs.csv", index_col = 0)
freq_auc2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_updated/AUCs/rf_aucs.csv", index_col = 0)
frequency_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/frequency_feature_info.csv", index_col = 0)
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results/"

###FREQUNECY BOXPLOTS
for val in frequency_info["Variable"].unique():
    freq_groups = frequency_info[frequency_info["Variable"] == val].index
    temp = freq_res1.loc[freq_groups,:]
    temp["group"] = frequency_info.loc[temp.index, "plot_name"]
    temp["ind"] = temp.index.values
    temp.index = temp["group"]
    temp = temp.iloc[[0,3,1,2],:]
    boxplotdata = np.vstack(freq_auc1.loc[temp["ind"], :].values)
    boxplotdata = pd.DataFrame(boxplotdata, index = temp["group"]).T
    g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
    plt.xlabel("AUC")
    plt.xlim(0.5, 1.0)
    plt.ylabel("")
    plt.title(val.replace("_", " ").capitalize())
    plt.tight_layout()
    plt.savefig(save_path + "BoxPlots/" + val + ".pdf", format='pdf')
    plt.show()

save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_updated/"

###FREQUNECY BOXPLOTS
for val in frequency_info["Variable"].unique():
    freq_groups = frequency_info[frequency_info["Variable"] == val].index
    temp = freq_res2.loc[freq_groups,:]
    temp["group"] = frequency_info.loc[temp.index, "plot_name"]
    temp["ind"] = temp.index.values
    temp.index = temp["group"]
    temp = temp.iloc[[0,3,1,2],:]
    boxplotdata = np.vstack(freq_auc2.loc[temp["ind"], :].values)
    boxplotdata = pd.DataFrame(boxplotdata, index = temp["group"]).T
    g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
    plt.xlabel("AUC")
    plt.xlim(0.5, 1.0)
    plt.ylabel("")
    plt.title(val.replace("_", " ").capitalize())
    plt.tight_layout()
    plt.savefig(save_path + "BoxPlots/" + val + ".pdf", format='pdf')
    plt.show()


#%%
    
    
bin_res1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results/rf_results.csv", index_col = 0)
bin_res2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_updated/rf_results.csv", index_col = 0)

bin_auc1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results/AUCs/rf_aucs.csv", index_col = 0)
bin_auc2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_updated/AUCs/rf_aucs.csv", index_col = 0)
feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results/"

inds = ["antiB_Week_cohort", "antiB_Month_cohort", "antiB_6Month_cohort", "antiB_Year_cohort"]

antib_res = bin_res1.loc[inds, :]
antib_res = pd.DataFrame(antib_res) #, index = feature_info.loc[antib_res.index, "plot_name"]
boxplotdata = np.vstack(bin_auc1.loc[antib_res.index, :].values)
boxplotdata = pd.DataFrame(boxplotdata, index = ["Week", "Month", "6 Month", "Year"]).T

g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.xlim(0.5, 1.0)
plt.ylabel("")
plt.title("Most recent antibiotic use")
plt.tight_layout()
plt.savefig(save_path + "BoxPlots/Antibiotic_use.pdf", format='pdf')
plt.show()




