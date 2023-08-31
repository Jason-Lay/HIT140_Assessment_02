import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math
import statsmodels.stats.weightstats as stm

# Define column headers
column_headers = ['ID','Jit_%','Jit_abs','Jit_r.a.p.','Jit_p.p.q.5','Jit_d.d.p.','Shim_%','Shim_abs','Shim_a.p.q.3',
                  'Shim_a.p.q.5','Shim_a.p.q.11','Shim_d.d.a.','Har_Autocorr','Har_NHR','Har_HNR','Pit_Median','Pit_Mean',
                  'Pit_Std_Dev','Pit_Min','Pit_Max','Pul_Num_Puls','Pul_Num_Prds','Pul_Mean_Prds','Pul_Std_Dev_Prds',
                  'Vo_Frac_Unvoiced','Vo_Num_Breaks','Vo_Degree_Breaks','UPDRS','PD']

# Read txt into a DataFrame
df = pd.read_csv('po1_data.txt',names=column_headers, delimiter=',')

# Compare distribution of features between PD and healthy groups
pd_group = df[df['PD'] == 1]
healthy_group = df[df['PD'] == 0]

# Create grouped bar plots for each feature
num_features = len(df.columns[1:28])
feature_names = df.columns[1:28]
x = range(num_features)
bar_width = 0.4

plt.figure(figsize=(12, 8))
plt.bar(x, pd_group[feature_names].mean(), bar_width, label='PD', align='center')
plt.bar([i + bar_width for i in x], healthy_group[feature_names].mean(), bar_width, label='Healthy', align='center')
plt.xticks([i + bar_width/2 for i in x], feature_names, rotation=90)
plt.title("Mean Feature Values for PD and Healthy Groups")
plt.xlabel("Feature")
plt.ylabel("Mean Value")
plt.legend()
plt.show()

# Confidence interval function for PD and non-PD
def cal_conf_int_statsmodels( x_bar, std, sample_size ):
    std_err = std / math.sqrt( sample_size )
    ci_low_stm, ci_upp_stm = stm._zconfint_generic(x_bar,std_err,alpha=0.05, alternative="two-sided")
    
    return ci_low_stm, ci_upp_stm

# Feature Selection and Inferential Analysis
significant_features = []

for col in df.columns[1:27]:
    t_stat, p_value = stats.ttest_ind(pd_group[col], healthy_group[col])
    if p_value < 0.05:
        significant_features.append(col)
        print(f"{col}: p-value = {p_value:.4f} (Significant)")
        conf_int_pd = cal_conf_int_statsmodels(pd_group[col].mean(), stats.tstd(pd_group[col]), len(pd_group[col]))
        conf_int_non_pd = cal_conf_int_statsmodels(healthy_group[col].mean(), stats.tstd(healthy_group[col]), len(healthy_group[col]))
        print(f"Confidence interval of '{col}' for PD group", conf_int_pd)
        print(f"Confidence interval of '{col}' for Healthy group", conf_int_non_pd, "\n")

# Plot the distribution of each significant feature
for feature in significant_features:
    plt.hist(pd_group[feature], bins=20, color='#3b75af', edgecolor='#3f75af', alpha=0.7, rwidth=0.8, label="PD (1)")
    plt.hist(healthy_group[feature], bins=20, color='#ef8636', edgecolor='#ef863f', alpha=0.7, rwidth=0.8, label="Non-PD (0)")
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f"Distribution of '{feature}' for PD and Healthy groups")
    plt.legend()
    plt.subplots_adjust(left=0.15)

    # add text for mean
    ax = plt.gca()
    plt.text(.8,.825, "µ1={:.3g}".format(pd_group[feature].mean()),
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes)
    plt.text(.8,.725, "µ0={:.3g}".format(healthy_group[feature].mean()),
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes)
    # add text for standard deviation
    plt.text(.8,.775, "σ1={:.3g}".format(stats.tstd(pd_group[feature])),
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes)
    plt.text(.8,.675, "σ0={:.3g}".format(stats.tstd(healthy_group[feature])),
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes)
    plt.show()

# Interpretation
print("Salient variables (features) that could be used to distinguish people with PD from healthy:")
for feature in significant_features:
    print(feature)
