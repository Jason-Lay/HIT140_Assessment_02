import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import math
import statsmodels.stats.weightstats as stm

# Define column headers
column_headers = ['ID','Jit_%','Jit_abs','Jit_r.a.p.','Jit_p.p.q.5','Jit_d.d.p.','Shim_%','Shim_abs','Shim_a.p.q.3',
                  'Shim_a.p.q.5','Shim_a.p.q.11','Shim_d.d.a.','Har_Autocorr','Har_NHR','Har_HNR','Pit_Median',
                  'Pit_Mean','Pit_Std_Dev','Pit_Min','Pit_Max','Pul_Num_Puls','Pul_Num_Prds','Pul_Mean_Prds',
                  'Pul_Std_Dev_Prds','Vo_Frac_Unvoiced','Vo_Num_Breaks','Vo_Degree_Breaks','UPDRS','PD']

# Read txt into a DataFrame
df = pd.read_csv('po1_data.txt',names=column_headers, delimiter=',')

# Split into PD and non-PD 
df_pd = df[df["PD"]==1]
df_non = df[df["PD"]==0]

#########################################################################################################################
##########         Functions to calculate t-statistics, P-value, mean, std, and confidence interval          ############
#########################################################################################################################
def calculate_tstats_pval( parkinson, non_parkinson ):
    # Calculate mean and standard deviation
    x_bar1 = st.tmean( parkinson )
    x_bar2 = st.tmean( non_parkinson )

    s1 = st.tstd( parkinson )
    s2 = st.tstd( non_parkinson )

    # sample size 
    n1 = len( parkinson )
    n2 = len( non_parkinson )

    t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1,
                                            x_bar2, s2, n2,
                                            equal_var="False", #different std 
                                            alternative="two-sided") 
    print(t_stats, p_val)
    
    
def cal_mean( column_numpy_array ):
    x_bar = st.tmean( column_numpy_array )
    return x_bar


def cal_std( column_numpy_array ):
    std = st.tstd( column_numpy_array )
    return std
    
    
def cal_tstats_pval( x_bar1, s1, n1, x_bar2, s2, n2):
    t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1,
                                            x_bar2, s2, n2,
                                            equal_var="False", #different std 
                                            alternative="two-sided")
    return t_stats, p_val

def cal_conf_int_statsmodels( x_bar, std, sample_size ):
    std_err = std / math.sqrt( sample_size )
    ci_low_stm, ci_upp_stm = stm._zconfint_generic(x_bar,std_err,alpha=0.05, alternative="two-sided")
    
    return ci_low_stm, ci_upp_stm


#########################################################################################################################
###################           CREATE LIST OF TITLES AND LABELS FOR EACH GRAPH           #################################
#########################################################################################################################
x_label_list = ["","Jitter Percentage (%)", "Absolute Jitter in Microseconds (μs)", "r.a.p", "p.p.q.5", "d.d.p", 
                "Shimmer Percentage (%)", "Absolute shimmer in decibals (db)", "a.p.q.3", "a.p.q.5", "a.p.q.11", "d.d.a",
                  "Harmonicity between NHR and HNR", "NHR", "HNR", "Median pitch", "Mean Pitch", "Standard Deviation of Pitch",
                    "Minimum pitch", "Maximum pitch", "Number of Pulses", "Number of Periods", "Mean Period",
                      "Standard Deviation of Period", "Fraction of Unvoiced Frames", "Number of Voice Breaks", "Degree of Voice Breaks"]
title_list = ["", "Jitter Percentage Distribution", "Absolute Jitter Distribution", "Jitter as Relative Amplitude Perturbation Distribution", 
              "Jitter as 5-point Period Perturbation Quotient Distribution", "D.D.P Distribution", "Shimmer Percentage Distribution",
                "Absolute Shimmer Distribution", "Shimmer as 3-point Amplitude Perturbation Quotient Distribution", 
                "Shimmer as 5-point Amplitude Perturbation Quotient Distribution", "Shimmer as 11-point Amplitude Perturbation Quotient Distribution",
                  "D.D.A Distribution", "Autocorrelation between NHR and HNR Distribution", "Noise-to-Harmonic Ratio Distribution", 
                  "Harmonic-to-Noise Ratio Distribution", "Median Pitch Distribution", "Mean Pitch Distribution",
                    "Standard Deviation of Pitch Distribution", "Minimum Pitch Distribution", "Maximum Pitch Distribution",
                      "Number of Pulses Distribution", "Number of Periods Distribution", "Mean Period Distribution", 
                      "Standard Deviation of Period Distribution", "Fraction of Unvoiced Frames Distribution",
                        "Number of Voice Breaks Distribution", "Degree of Voice Breaks Distribution"]

#########################################################################################################################
###################           Function to Plot Histrograms for PD and Non-PD            #################################
#########################################################################################################################
def plot_graph( pd, non_pd, x_label, title, mean1, mean2, std1, std2): # require title, mean1, mean2
    # plot both histograms
    plt.hist( pd, bins=20, edgecolor="red",alpha=0.7, rwidth=0.8, label="PD (1)")
    plt.hist( non_pd, bins=20, facecolor='orange', edgecolor="yellow",alpha=0.8, rwidth=0.8, label="Non-PD (0)")
    # label axis
    plt.xlabel( x_label )
    plt.ylabel('Frequency')

    # label title and legends
    plt.legend()
    plt.title(title)
    plt.subplots_adjust(left=0.15)
    # add text for mean
    ax = plt.gca()
    plt.text(.8,.825, "µ1={:.3g}".format(mean1),
             horizontalalignment="left",
             verticalalignment="top",
             transform=ax.transAxes)
    plt.text(.8,.725, "µ0={:.3g}".format(mean2),
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes)
    # add text for standard deviation
    plt.text(.8,.775, "σ1={:.3g}".format(std1),
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes)
    plt.text(.8,.675, "σ0={:.3g}".format(std2),
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes)
    plt.show()

def null_hyp_test( pval ):
    if pval < 0.05:
        print("\n We reject the null hypothesis")
    else:
        print("\n We accept the null hypothesis")

#########################################################################################################################
##############          Loop through all columns to plot and calculate relevant data in histogram          ##############
#########################################################################################################################
# Plot histograms for PD and Non PD of column 1 "jitter_percentage" to column 26 "voice_degree_of_voice_breaks"
# Run all the relevant calculations for each graph 
for i in range(1,27):
    # select columns individually as it loops
    column_range = df_pd.columns[i:i + 1]
    # Assign current column for PD and non PD in the form of a Numpy Array to its own variable
    parkinson = df_pd[column_range].to_numpy()
    non_parkinson = df_non[column_range].to_numpy()
    
    # assign mean to variables
    mean_pd = cal_mean(parkinson)
    mean_non_pd = cal_mean(non_parkinson)
    
    # assign standard deviation to variables
    std_pd = cal_std(parkinson)[0]
    std_non_pd = cal_std(non_parkinson)[0]
    
    # t-statistics
    t_stat = cal_tstats_pval( mean_pd,
                           cal_std(parkinson),
                           len(parkinson),
                            mean_non_pd,
                            cal_std(non_parkinson),
                            len(non_parkinson))[0][0]
    # P-value
    p_val = cal_tstats_pval( mean_pd,
                           cal_std(parkinson),
                           len(parkinson),
                            mean_non_pd,
                            cal_std(non_parkinson),
                            len(non_parkinson))[1][0]

    # Confidence interval for PD and non-PD
    conf_int_pd = cal_conf_int_statsmodels( mean_pd, std_pd, len(parkinson) )
    conf_int_non_pd = cal_conf_int_statsmodels( mean_non_pd, std_non_pd, len(non_parkinson) )
    
    
    if i > 1: 
        print("\n")

    print(title_list[i].upper() + ": ")
    
    null_hyp_test( p_val )

    print("This is the mean for pd: %.3g" % mean_pd)
    print("This is the mean for non-pd: %.3g" % mean_non_pd)
    
    print("this is the standard deviation for pd: %.3g" % std_pd)
    print("this is the standard deviation for non-pd: %.3g" % std_non_pd)
    
    print("This is confidence interval for pd: ", conf_int_pd)
    print("This is confidence interval for non-pd: ", conf_int_non_pd)
    
    print("This is the t-statistics and p-value: ", t_stat, p_val)

    # calculate_tstats_pval( parkinson, non_parkinson )
    plot_graph( parkinson, non_parkinson, x_label_list[i], title_list[i], mean_pd, mean_non_pd, std_pd, std_non_pd)
    
