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
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot  


metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
file_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/AGP_File_Metadata.csv", index_col = 0)
otufull_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Raw_Data/AGP_merged2_s50_rar10k_forR_md5.txt", sep = "\t").T
#agp_healthy_cohort = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Feature_cohorts/agp_healthy_cohort.csv", index_col = 0)


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

metadata_df = metadata_df[(metadata_df["age_years"] >= 20.0) & 
                          (metadata_df["age_years"] <= 80.0) &
                          (metadata_df["antibiotic_history"].isin(["Year", "I have not taken antibiotics in the past year."])) &
                          (metadata_df["ibd"] == 0) &
                          (metadata_df["diabetes"] == 0) &
                          (metadata_df["bmi"] >= 12.5) & 
                          (metadata_df["bmi"] <= 40.0)]

print "Metadata selection Poputlation (unmatched):", len(metadata_df)


metadata_df = metadata_df[(metadata_df["alcohol_frequency"] != 5) &
                          (metadata_df["milk_cheese_frequency"] !=  5) &
                          (metadata_df["meat_eggs_frequency"] != 5) &
                          (metadata_df["vegetable_frequency"] != 5) &
                          (~metadata_df["bowel_movement_quality"].isin(["Unspecified", "Not provided", "I don't know, I do not have a point of reference"]))]


print "Metadata selection Poputlation (matched):", len(metadata_df)

#%%


no_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 0]["age_years"]
rare_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 1]["age_years"]
occ_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 2]["age_years"]
reg_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 3]["age_years"]
day_alc = metadata_df_healthy[metadata_df_healthy["alcohol_frequency"] == 4]["age_years"]
full_bins = len(metadata_df_healthy["age_years"].unique()) + 1
full_bins = 96

plt.figure(figsize = (10, 7))
plt.hist([day_alc, reg_alc, occ_alc, rare_alc, no_alc],
         color = ["blue", "green", "yellow", "orange", "red"],
         label = ["Daily", "Regularly", "Occasionally", "Rarely", "Never"],
         bins = full_bins, stacked=True, density=True)
plt.legend()
plt.title("Alcohol consumption frequency in healthy population")
plt.xlabel("Age")
#plt.tight_layout()
plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Figs/alcohol_freq_hist.pdf")
plt.show()


#%%

under = metadata_df_healthy[(metadata_df_healthy["bmi"] < 18.5) & (metadata_df_healthy["bmi"] >= 12.5)]["age_years"]
norm = metadata_df_healthy[(metadata_df_healthy["bmi"] < 25.0) & (metadata_df_healthy["bmi"] >= 18.5)]["age_years"]
over = metadata_df_healthy[(metadata_df_healthy["bmi"] < 30.0) & (metadata_df_healthy["bmi"] >= 25.0)]["age_years"]
obese = metadata_df_healthy[(metadata_df_healthy["bmi"] <= 40.0) & (metadata_df_healthy["bmi"] >= 30.0)]["age_years"]

full_bins = len(metadata_df_healthy["age_years"].unique()) + 1
full_bins = 96

plt.figure(figsize = (10, 7))
plt.hist([under, norm, over, obese],
         color = ["red", "yellow", "green", "blue"],
         label = ["Underweight", "Normal weight", "Overweight", "Obese"],
         bins = full_bins, stacked=True, density=True)
plt.legend()
plt.title("BMI categories in healthy population")
plt.xlabel("Age")
#plt.tight_layout()
plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Figs/bmi_cat_hist.pdf")
plt.show()



#%%

agp_healthy_cohort["index"] = agp_healthy_cohort.index.values
agp_healthy_cohort["text"] = agp_healthy_cohort["index"] + ": " + agp_healthy_cohort["sex"] + ", " + agp_healthy_cohort["age_years"].astype(str)

fig = go.Figure(data=go.Scattergeo(lon = agp_healthy_cohort.loc[:, "longitude"], 
                                   lat = agp_healthy_cohort.loc[:, "latitude"],
                                   text = agp_healthy_cohort["text"],
                                   mode = 'markers',
                                   marker = dict(size = 3,
                                                 opacity = 0.8,
                                                 reversescale = True,
                                                 autocolorscale = False,
                                                 symbol = 'circle',
                                                 line = dict(width=1, color='rgba(102, 102, 102)'),
                                                 colorscale = 'Blues',
                                                 cmin = 1,
                                                 color = agp_healthy_cohort['age_years'],
                                                 cmax = agp_healthy_cohort['age_years'].max(),
                                                 colorbar_title="Participant age" )))
fig.update_layout(geo = dict(showland = True,
                             landcolor = "rgb(212, 212, 212)",
                             subunitcolor = "rgb(255, 255, 255)",
                             countrycolor = "rgb(255, 255, 255)",
                             showlakes = True,
                             lakecolor = "rgb(255, 255, 255)",
                             showsubunits = True,
                             showcountries = True,
                             resolution = 110,
                             projection = dict(type = 'mercator', rotation_lon = -100 ),
                             lonaxis = dict( showgrid = True,  gridwidth = 0.5, range= [ -170.0, 25.0 ], dtick = 5),
                             lataxis = dict (showgrid = True, gridwidth = 0.5, range= [ 16.0, 68.0 ], dtick = 5)),
                title='AGP Healthy Participants',)
plot(fig)

#%%

bin_res1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_no_matching/rf_results.csv", index_col = 0)
bin_res2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_standard/rf_results.csv", index_col = 0)

bin_auc1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_no_matching/AUCs/rf_aucs.csv", index_col = 0)
bin_auc2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_standard/AUCs/rf_aucs.csv", index_col = 0)

feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)


standard_match_vars = ["alcohol_rare_cohort","Underweight_cohort", "alcohol_regular_cohort", "Obese_cohort", "vegetarian_cohort",
                       "pescatarian_cohort","diabetes_typeII_cohort","ibs","Overweight_cohort", "age_1-2"
                       'alcohol_occasional_cohort','antiB_6Month_cohort', u'antiB_Month_cohort',
                       u'antiB_Week_cohort', u'antiB_Year_cohort', "sex", "age_over70", "gender_healthy_cohort", "IBD_cohort"]

temp = bin_res1[((bin_res1["p_val"] <= 0.05) & (bin_res1["auc_median"] >= 0.7)) | bin_res1.index.isin(standard_match_vars)].sort_values("auc_median", ascending = False)
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

auc_change = bin_res2["auc_mean"] - bin_res1["auc_mean"]
auc_change = pd.DataFrame(auc_change)

auc_change["auc_matched"] = bin_res2["auc_mean"]
auc_change["auc_unmatched"] = bin_res1["auc_mean"]

auc_change["pval_matched"] = bin_res2["p_val"]
auc_change["pval_unmatched"] = bin_res1["p_val"]

auc_change["abs_delta_auc"] = auc_change["auc_mean"].abs()
auc_change.sort_values(by = ["abs_delta_auc"], inplace = True, ascending = False)
auc_change.rename(columns = {"auc_mean": "delta_auc"}, inplace = True)
auc_change.to_csv("/Users/sklarjg/Desktop/auc_matching_change.csv")
#%%

sig_vars = auc_change[((auc_change["pval_unmatched"] <= 0.05) & (auc_change["auc_unmatched"] >= 0.65)) | ((auc_change["pval_matched"] <= 0.05) & (auc_change["auc_matched"] >= 0.65))].index.values
auc_change = auc_change.loc[sig_vars, :]


plt.figure(figsize = (5, 10))
auc_change = auc_change.iloc[:25, :]
auc_change.sort_values(by = ["abs_delta_auc"], inplace = True, ascending = False)
auc_change.index = feature_info.loc[auc_change.index, "plot_name"]
auc_change["delta_auc"].plot.barh(color='k', alpha=0.5)
plt.xlabel("Mean-AUC change after new matching")
#plt.tight_layout()
plt.savefig("/Users/sklarjg/Desktop/auc_matching_change.pdf")
plt.show()


#%%
bin_res = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_test/rf_results.csv", index_col = 0)
bin_auc = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_test/AUCs/rf_aucs.csv", index_col = 0)

file_list = ["diabetes_cohort_no_matched", "diabetes_cohort_age", "diabetes_cohort_alc", 
             "diabetes_cohort_bmi",  "diabetes_cohort_diet", "diabetes_cohort_matched", "diabetes_cohort_sex"]

file_names = ["Diabetes Type II [no matching]","Diabetes Type II [age]", "Diabetes Type II [alcohol consumption]", 
              "Diabetes Type II [BMI]", "Diabetes Type II [diet type]", "Diabetes Type II", "Diabetes Type II [sex]"]

feature_info = pd.DataFrame({"file_name": file_list, "plot_name": file_names})
feature_info["type"] = "disease"
feature_info = feature_info.set_index("file_name")

temp = bin_res.sort_values("auc_median", ascending = False)

boxplotdata = bin_auc.loc[temp.index, :].values
boxplotdata = pd.DataFrame(boxplotdata, index = feature_info.loc[temp.index, "plot_name"]).T
sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.xlim(0.5, 1.0)
plt.ylabel("")
plt.tight_layout()
plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_test/auc_dists.pdf")
plt.show()
    
#%%

feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)
save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results/"

inds = ["antiB_Week_cohort", "antiB_Month_cohort", "antiB_6Month_cohort", "antiB_Year_cohort"]

antib_res = bin_res2.loc[inds, :]
antib_res = pd.DataFrame(antib_res) #, index = feature_info.loc[antib_res.index, "plot_name"]
boxplotdata = np.vstack(bin_auc2.loc[antib_res.index, :].values)
boxplotdata = pd.DataFrame(boxplotdata, index = ["Week", "Month", "6 Month", "Year"]).T

g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.xlim(0.5, 1.0)
plt.ylabel("")
plt.title("Most recent antibiotic use")
plt.tight_layout()
save_path = "/Users/sklarjg/Desktop/"
plt.savefig(save_path + "Antibiotic_use_matched.pdf", format='pdf')
plt.show()


inds = ["alcohol_daily_cohort",  "alcohol_regular_cohort", "alcohol_occasional_cohort", "alcohol_rare_cohort"]

antib_res = bin_res2.loc[inds, :]
antib_res = pd.DataFrame(antib_res) #, index = feature_info.loc[antib_res.index, "plot_name"]
boxplotdata = np.vstack(bin_auc2.loc[antib_res.index, :].values)
boxplotdata = pd.DataFrame(boxplotdata, index = ["Daily", "Regularly", "Occasionally", "Rarely"]).T

g = sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
plt.xlabel("AUC")
plt.xlim(0.5, 1.0)
plt.ylabel("")
plt.title("Most recent antibiotic use")
plt.tight_layout()
save_path = "/Users/sklarjg/Desktop/"
plt.savefig(save_path + "Alcohol_use_matched.pdf", format='pdf')
plt.show()

#%%
temp_sig = bin_res2[bin_res2["p_val"] <= 0.05]
temp_not_sig = bin_res2[bin_res2["p_val"] > 0.05]

#temp_sig.loc[temp_sig["n_samples"] > 2000, "n_samples"] = 2000
#temp_not_sig.loc[temp_not_sig["n_samples"] > 2000, "n_samples"] = 2000

not_sig = temp_not_sig["p_val"]*300
sig = temp_sig["p_val"]*300

plt.scatter(temp_not_sig["n_samples"], temp_not_sig["auc_std"], c = "black", s = not_sig)
plt.scatter(temp_sig["n_samples"], temp_sig["auc_std"], c = "red", s = sig)

plt.xlabel("cohort size")
plt.ylabel("AUC Standard Deviation")
plt.savefig("/Users/sklarjg/Desktop/cohortsize_aucstd.pdf")
plt.show()

plt.scatter(temp_not_sig["n_samples"], temp_not_sig["auc_mean"], c = "black", s = not_sig)
plt.scatter(temp_sig["n_samples"], temp_sig["auc_mean"], c = "red", s = sig)

plt.xlabel("cohort size")
plt.ylabel("AUC Mean")
plt.savefig("/Users/sklarjg/Desktop/cohortsize_aucmean.pdf")
plt.show()

temp_sig = bin_res1[bin_res1["p_val"] <= 0.05]
temp_not_sig = bin_res1[bin_res1["p_val"] > 0.05]

#temp_sig.loc[temp_sig["n_samples"] > 2000, "n_samples"] = 2000
#temp_not_sig.loc[temp_not_sig["n_samples"] > 2000, "n_samples"] = 2000

not_sig = temp_not_sig["p_val"]*300
sig = temp_sig["p_val"]*300

plt.scatter(temp_not_sig["n_samples"], temp_not_sig["auc_std"], c = "black", s = not_sig)
plt.scatter(temp_sig["n_samples"], temp_sig["auc_std"], c = "red", s = sig)

plt.xlabel("cohort size")
plt.ylabel("AUC Standard Deviation")
plt.savefig("/Users/sklarjg/Desktop/cohortsize_aucstd_unmatched.pdf")
plt.show()

plt.scatter(temp_not_sig["n_samples"], temp_not_sig["auc_mean"], c = "black", s = not_sig)
plt.scatter(temp_sig["n_samples"], temp_sig["auc_mean"], c = "red", s = sig)

plt.xlabel("cohort size")
plt.ylabel("AUC Mean")
plt.savefig("/Users/sklarjg/Desktop/cohortsize_aucmean_unmatched.pdf")
plt.show()

#%%
freq_res1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_no_matching/rf_results.csv", index_col = 0)
freq_res2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_standard/rf_results.csv", index_col = 0)

freq_auc1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_no_matching/AUCs/rf_aucs.csv", index_col = 0)
freq_auc2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_standard/AUCs/rf_aucs.csv", index_col = 0)
frequency_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/frequency_feature_info.csv", index_col = 0)

save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_no_matching/"

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

save_path = "/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/frequency_results_standard/"

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





