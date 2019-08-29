#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:03:31 2019

@author: sklarjg
"""

import pandas as pd
import matplotlib.pyplot as plt

bin_res1 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_no_matching/rf_results.csv", index_col = 0)
bin_res2 = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Results/binary_results_standard/rf_results.csv", index_col = 0)
feature_info = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/feature_info.csv", index_col = 0)

auc_change = bin_res2["auc_mean"] - bin_res1["auc_mean"]
auc_change = pd.DataFrame(auc_change)
auc_change["change_abs"] = auc_change.abs()
sig_vars = bin_res1[(bin_res1["p_val"] <= 0.05) & (bin_res1["auc_mean"] >= 0.65)].index.values
auc_change = auc_change.loc[sig_vars, :]


plt.figure(figsize = (5, 10))

auc_change.sort_values(by = ["change_abs"], inplace = True, ascending = False)
auc_change = auc_change.iloc[:20, :]
auc_change.sort_values(by = ["auc_mean"], inplace = True, ascending = False)
auc_change.index = feature_info.loc[auc_change.index, "plot_name"]
auc_change["auc_mean"].plot.barh(color='k', alpha=0.5)
plt.xlabel("Mean-AUC change after new matching")
#plt.tight_layout()
plt.savefig("/Users/sklarjg/Desktop/auc_matching_change.pdf")
plt.show()

#%%

compare = pd.DataFrame({"sampes": bin_res1["n_samples"], 
                        "mean_1": bin_res1["auc_mean"], 
                        "pval_1": bin_res1["p_val"],
                        "mean_2": bin_res2["auc_mean"], 
                        "pval_2": bin_res2["p_val"],
                        "mean_delta": bin_res2["auc_mean"] - bin_res1["auc_mean"]})
compare = compare[(compare["pval_1"] <= 0.05) | (compare["pval_2"] <= 0.05)]
compare.sort_values(by = ["mean_delta"], inplace = True)



#%%

'''
metadata_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata.csv", index_col = 0)
metadata_df2 = metadata_df[~metadata_df["age_years"].isin(["Not provided", "Unspecified"])]
metadata_df2["age_years"] = metadata_df2["age_years"].astype(float)
infants_df = metadata_df2[metadata_df2["age_years"] <= 10]
infant_drinkers = infants_df[infants_df["alcohol_frequency"] != 0].index.values
metadata_df.loc[infant_drinkers, "alcohol_frequency"] = 0
metadata_df.to_csv("/Users/sklarjg/Desktop/MICROBIOME/AmericanGutProj/Data/Cleaned_data/AGP_Metadata2.csv")

'''