import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import math
from pandas.api.types import CategoricalDtype
from scipy import stats
from scipy.stats import linregress
from sklearn.utils import resample
import pdb
from sklearn.metrics import r2_score

# found dixon's test at https://sebastianraschka.com/Articles/2014_dixon_test.html#implementing-a-dixon-q-test-function
# no need to re-invent the wheel

# dictionary to look up the critical Q-values (dictionary values)
# for different sample sizes 3+

q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
       0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
       0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
       0.277, 0.273, 0.269, 0.266, 0.263, 0.26
      ]

q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
       0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
       0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
       0.308, 0.305, 0.301, 0.29
      ]

q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
       0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
       0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
       0.384, 0.38, 0.376, 0.372
       ]

Q90 = {n:q for n,q in zip(range(3,len(q90)+1), q90)}
Q95 = {n:q for n,q in zip(range(3,len(q95)+1), q95)}
Q99 = {n:q for n,q in zip(range(3,len(q99)+1), q99)}

def dixon_test(data, left=True, right=True, q_dict=Q95):
    """
    Keyword arguments:
        data = A ordered or unordered list of data points (int or float).
        left = Q-test of minimum value in the ordered list if True.
        right = Q-test of maximum value in the ordered list if True.
        q_dict = A dictionary of Q-values for a given confidence level,
            where the dict. keys are sample sizes N, and the associated values
            are the corresponding critical Q values. E.g.,
            {3: 0.97, 4: 0.829, 5: 0.71, 6: 0.625, ...}

    Returns a list of 2 values for the outliers, or None.
    E.g.,
       for [1,1,1] -> [None, None]
       for [5,1,1] -> [None, 5]
       for [5,1,5] -> [1, None]

    """
    assert(left or right), 'At least one of the variables, `left` or `right`, must be True.'
    assert(len(data) >= 3), 'At least 3 data points are required'
    assert(len(data) <= max(q_dict.keys())), 'Sample size too large'

    sdata = sorted(data)
    Q_mindiff, Q_maxdiff = (0,0), (0,0)

    if left:
        Q_min = (sdata[1] - sdata[0])
        try:
            Q_min /= (sdata[-1] - sdata[0])
        except ZeroDivisionError:
            pass
        Q_mindiff = (Q_min - q_dict[len(data)], sdata[0])

    if right:
        Q_max = abs((sdata[-2] - sdata[-1]))
        try:
            Q_max /= abs((sdata[0] - sdata[-1]))
        except ZeroDivisionError:
            pass
        Q_maxdiff = (Q_max - q_dict[len(data)], sdata[-1])

    if not Q_mindiff[0] > 0 and not Q_maxdiff[0] > 0:
        outliers = [None, None]

    elif Q_mindiff[0] == Q_maxdiff[0]:
        outliers = [Q_mindiff[1], Q_maxdiff[1]]

    elif Q_mindiff[0] > Q_maxdiff[0]:
        outliers = [Q_mindiff[1], None]

    else:
        outliers = [None, Q_maxdiff[1]]

    return outliers

def get_pass_dixonsq(df_in, groupby_list):
    '''get the whether the observation pass dixons'q or not
    Parameters
    ----------
    df: pandas dataframe
         contains relevant information to test triplicates
    groupby_list: list
        list of variables to be grouped by
    Returns
    -------
    df: pandas dataframe
         pass_dixonsq added where needed
    '''
    df = df_in.copy() # fixes pandas warnings
    df["pass_dixonsq"] = 1
    df.loc[df[df.Cq.isna()].index, 'pass_dixonsq'] = 0

    for _, data in df.groupby(groupby_list):
        # slice out triplicate values
        tri_vals = data['Cq_copy']

        # assigning the ones that don't pass dixonsq to be 0
        # if there are less than 3 values, automatically keeping them
        if len(tri_vals) == 3:
            dt_vals = dixon_test(tri_vals)
            if dt_vals != [None, None]:
                if None in dt_vals:
                    dt_vals.remove(None)
                for outlier in dt_vals:
                    df.loc[data[data.Cq_copy == float(outlier)].index, 'pass_dixonsq'] = 0
    return(df)

def get_cq_mean(df_in, groupby_list, use_dixonsq=False):
    '''get the mean of Cq triplicates

    Parameters
    ----------
    df: pandas dataframe
         contains relevant information to calculate the mean from Cq

    groupby_list: list
        list of variables to be grouped by

    use_dixonsq: boolean
        if true, mean is calculated accounting for dixonsq with Cq

    Returns
    -------
    df: pandas dataframe
         Cq_mean added and propagated
    '''
    df = df_in.copy() # fixes pandas warnings

    if use_dixonsq == True:
      #if pass_dixonsq is 0, convert Cq_copy to nan so not used in mean calc
      df.loc[df.pass_dixonsq==0, 'Cq_copy'] = np.nan

    mean_df = df.groupby(groupby_list)['Cq_copy'].apply(np.mean)
    mean_df = pd.DataFrame(mean_df)
    mean_df = mean_df.rename(columns={"Cq_copy": "Cq_mean"})
    mean_df.reset_index(inplace=True)
    df = pd.merge(df, mean_df)
    return(df)

def compute_linear_info(plate_data):
    '''compute the information for linear regression

    Parameters
    ----------
    plate_data: pandas dataframe with columns
      Cq_mean (already processed to remove outliers)
      log_Quantity
    Returns
    -------
    slope, intercept, r2 and efficiency
    '''
    y = plate_data['Cq_mean']
    x = plate_data['log_Quantity']
    model = np.polyfit(x, y, 1)
    predict = np.poly1d(model)
    r2 = r2_score(y, predict(x))
    slope, intercept = model
    efficiency = (10**(-1/slope)) - 1

    # abline_values = [slope * i + intercept for i in x]
    return(slope, intercept, r2, efficiency)#, abline_values])

def combine_triplicates(plate_df_in, checks_include):
    '''
    Flag outliers via Dixon's Q
    Flag outliers with another method
    Calculate the Cq means after removing outliers
    TODO figure out if there's a better way to remove outliers
    TODO this function should collapse triplicates down to samples with Cq_means
    it should add a column saying how many replicates were used
    but we are keeping all triplicates to check for errors right now

    Params
    plate_df_in:
        qpcr data in pandas df, must be 1 plate with 1 target
        should be in the format from QuantStudio3 with
        columns 'Target', 'Sample', 'Cq'
    checks_include:
        which way to check for outliers options are
        ('all', 'dixonsq_only', 'std_check_only', None)
    Returns
    plate_df: same data, with additional columns
        pass_dixonsq (0 or 1)
        Cq_mean (calculated mean of Cq after excluding outliers from dixonsq)
    '''

  if (checks_include not in ['all', 'dixonsq_only', 'std_check_only', None]):
    raise ValueError('''invalid input, must be one of the following: 'all',
                      'dixonsq_only', 'std_check_only'or None''')

  plate_df = plate_df_in.copy() # fixes pandas warnings

  # make copy of Cq column and turn this to np.nan for outliers
  plate_df['Cq_copy'] = plate_df['Cq'].copy()

  # check if NTCs amplified and flag
  # assert qpcr_test[qpcr_test.Sample=='NTC']['Cq'].isna().all()

  # Test triplicates with Dixon's Q
  use_dixonsq = False
  if checks_include in ['all', 'dixonsq_only']:
    use_dixonsq = True
  plate_df = get_pass_dixonsq(plate_df,["Target", "Sample"])

  # convert anything failing the outlier test(s) to np.nan
  # summarize to get mean, std, counts with and without outliers removed

  plate_df = get_cq_mean(plate_df, ["Target", "Sample"], use_dixonsq=use_dixonsq)

  return(plate_df)

def process_standard(plate_df):
    '''
    from single plate with single target, calculate standard curve
    TODO require at least 2 triplicates or else convert to na
    Params:
        plate_df: output from combine_triplicates(); df containing Cq_mean
        must be single plate with single target
    Returns
        num_points: number of points used in new std curve
        lowest_pt: the value of the lowest pt used in the new std curve
        slope:
        intercept:
        r2:
        efficiency:
    '''
  if len(plate_df.Target.unique()) > 1:
    raise ValueError('''More than one target in this dataframe''')
  standard_df = plate_df[plate_df.Task == 'Standard'].copy()

  standard_df['log_Quantity'] = standard_df.apply(lambda row: np.log10(pd.to_numeric(row.Quantity)), axis = 1)
  std_curve_df = standard_df[['Cq_mean', 'log_Quantity']].drop_duplicates().dropna()
  num_points = std_curve_df.shape[0]

  lowest_pt = np.nan
  slope, intercept, r2, efficiency = (np.nan, np.nan, np.nan, np.nan)

  if num_points > 3:
    lowest_pt = 10**min(standard_df.log_Quantity)
    slope, intercept, r2, efficiency = compute_linear_info(std_curve_df)

  return(num_points, lowest_pt, slope, intercept, r2, efficiency)

def process_unknown(plate_df, std_curve_info):
    '''
    Calculates quantity based on Cq_mean and standard curve
    Params
        plate_df: output from combine_triplicates(); df containing Cq_mean
        must be single plate with single target
        std_curve_info: output from process_standard() as a list
    Returns
        unknown_df: the unknown subset of plate_df, with 2 new columns
        (Quantity_recalc, and q_diff)
        these columns represent the recalculated quantity using Cq mean and the
        slope and intercept from the std curve
    '''


    [num_points, lowest_pt, slope, intercept, r2, efficiency] = std_curve_info
    unknown_df = plate_df[plate_df.Task == 'Unknown'].copy()
    unknown_df['Quantity_recalc'] =np.nan
    unknown_df['q_diff'] =np.nan
    unknown_df['Quantity_recalc'] = 10**((unknown_df['Cq_mean'] - intercept)/slope)
    unknown_df.loc[unknown_df[unknown_df.Cq_copy == 0].index, 'Quantity_recalc'] = np.nan
    unknown_df['q_diff'] = unknown_df['Quantity'] - unknown_df['Quantity_recalc']
    return(unknown_df)

def process_qpcr_raw(qpcr_raw):
  '''wrapper to process whole sheet at once by plate_id and Target'''
  std_curve_df = []
  qpcr_processed = []

  for [plate_id, target], df in qpcr_raw.groupby(["plate_id", "Target"]):
    no_outliers_df = combine_triplicates(df, 'all')

    # define outputs and fill with default values
    num_points, lowest_pt, slope, intercept, r2, efficiency = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    unknown_df = df[df.Task == 'Unknown']

    # if there are >3 pts in std curve, calculate stats and recalculate quants
    num_points = df[df.Task == 'Standard'].drop_duplicates('Sample').shape[0]
    if num_points > 3:
      num_points, lowest_pt, slope, intercept, r2, efficiency = process_standard(no_outliers_df)
      std_curve_info = [num_points, lowest_pt, slope, intercept, r2, efficiency]
      unknown_df = process_unknown(no_outliers_df, std_curve_info)
    std_curve_df.append([plate_id, target, num_points, lowest_pt, slope, intercept, r2, efficiency])
    qpcr_processed.append(unknown_df)


  # compile into dataframes
  std_curve_df = pd.DataFrame.from_records(std_curve_df, columns = ['plate_id', 'target', 'num_points', 'lowest_pt', 'slope', 'intercept', 'r2', 'efficiency'])
  qpcr_processed = pd.concat(qpcr_processed)
  qpcr_processed = qpcr_processed.merge(std_curve_df, how='left', on='plate_id')

  return(qpcr_processed, std_curve_df)
