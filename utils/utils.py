import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def perform_anova(df: pd.DataFrame, market_id: int) -> pd.DataFrame:
    """
    Performs ANOVA for a specified MarketID to compare the mean sales among different promotions.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        market_id (int): The MarketID for which to perform the ANOVA test.

    Returns:
        pd.DataFrame: A DataFrame containing the ANOVA results or an error message.
    """
    market_df = df[df["MarketID"] == market_id]
    model = ols('SalesInThousands ~ C(Promotion)', data=market_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def perform_tukey_test(df: pd.DataFrame, market_id: int) -> None:
    """
    Performs Tukey's HSD test for a specified MarketID to compare the mean sales among different promotions.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        market_id (int): The MarketID for which to perform the Tukey HSD test.
    """
    market_df = df[df["MarketID"] == market_id]
    if len(market_df["Promotion"].unique()) > 1:
        tukey = pairwise_tukeyhsd(endog=market_df['SalesInThousands'], groups=market_df['Promotion'], alpha=0.05)
        print(f"Tukey HSD Test Results for MarketID {market_id}")
        print(tukey)
        tukey.plot_simultaneous()
        plt.title(f'Tukey HSD Test Results for MarketID {market_id}')
        plt.show()
        
        best_promotion = market_df.groupby('Promotion')['SalesInThousands'].mean().idxmax()
    else:
        print(f"MarketID {market_id} has insufficient groups for Tukey HSD test")
        
def identify_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Identifies outliers in a specified column of a DataFrame using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column in which to identify outliers.

    Returns:
        pd.DataFrame: A DataFrame containing the rows identified as outliers in the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def analyze_market_size(df: pd.DataFrame, market_size: str) -> None:
    """
    Analyzes the effect of promotions on sales within a specified market size.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        market_size (str): The market size to analyze ('Small', 'Medium', 'Large').
    """
    df_filtered = df[df['MarketSize'] == market_size]
    
    model = ols('SalesInThousands ~ C(Promotion)', data=df_filtered).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    print(f"ANOVA Results for {market_size} Market Size")
    print(anova_table)
    
    if anova_table['PR(>F)'][0] < 0.05:
        print(f"There is a statistically significant difference in mean sales among the promotions for {market_size} market size.")
        tukey = pairwise_tukeyhsd(endog=df_filtered['SalesInThousands'], groups=df_filtered['Promotion'], alpha=0.05)
        print(tukey)
        tukey.plot_simultaneous()
        plt.title(f'Tukey HSD Test Results for {market_size} Market Size')
        plt.show()
    else:
        print(f"There is no statistically significant difference in mean sales among the promotions for {market_size} market size.")
        
def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean and confidence interval for a given dataset.

    Args:
        data (array-like): A list or array of numerical data points.
        confidence (float, optional): The confidence level for the interval. Defaults to 0.95.

    Returns:
        tuple: A tuple containing the mean, lower bound of the confidence interval, and upper bound of the confidence interval.
    """
    n = len(data)
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def bootstrap_mean(data, n_bootstrap=1000):
    """
    Generate bootstrap samples to estimate the distribution of the sample mean.

    Args:
        data (array-like): The input data from which to generate bootstrap samples.
        n_bootstrap (int, optional): The number of bootstrap samples to generate. Defaults to 1000.

    Returns:
        list: A list of mean values from the bootstrap samples.
    """
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    return means

def bootstrap_confidence_interval(data1, data2, n_bootstrap=1000, ci=95):
    """
    Estimate the confidence interval for the difference in means between two datasets using bootstrap sampling.

    Args:
        data1 (array-like): The first dataset.
        data2 (array-like): The second dataset.
        n_bootstrap (int, optional): The number of bootstrap samples to generate. Defaults to 1000.
        ci (int, optional): The confidence level for the interval (e.g., 95 for 95% confidence interval). Defaults to 95.

    Returns:
        tuple: The lower and upper bounds of the confidence interval for the mean difference.
    """
    mean_diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        mean_diff = np.mean(sample1) - np.mean(sample2)
        mean_diffs.append(mean_diff)
    lower_bound = np.percentile(mean_diffs, (100-ci)/2)
    upper_bound = np.percentile(mean_diffs, 100 - (100-ci)/2)
    return lower_bound, upper_bound

def calculate_effect_size(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.

    Args:
        group1 (array-like): First group of data.
        group2 (array-like): Second group of data.

    Returns:
        float: The calculated effect size (Cohen's d).
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / (len(group1) + len(group2) - 2))
    return (mean1 - mean2) / pooled_std