import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import math
from pandas.api.types import CategoricalDtype
from scipy import stats as sci
from scipy.stats import linregress
from scipy.stats.mstats import gmean
from scipy.stats import gstd
from sklearn.utils import resample
import pdb
from sklearn.metrics import r2_score
from statistics import median
#grubbs test package
import outliers
from outliers import smirnov_grubbs as grubbs

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

def median_test(Cqs, max_spread = 0.5):
  '''
  for a group of Cqs, test whether each
  is within max_spread of the median

  params
  Cqs: pandas series of Cts
  max_spread: defaults to 0.5 Ct

  result: list of booleans, same length as Cqs
  '''
  num_points = len(Cqs[~Cqs.isna()])
  if num_points > 2:
    m = median(Cqs[~Cqs.isna()])
    median_dist = [abs(i-m) <= max_spread for i in Cqs]

  #median of 2 numbers will split between them, half max_spread
  elif num_points == 2:
    m = median(Cqs[~Cqs.isna()])
    median_dist = [abs(i-m) <= (max_spread / 2)  for i in Cqs]

  #no median_test can be done
  elif num_points < 2:
    median_dist = [False  for i in Cqs]
  return(median_dist)

def test_median_test():
  t1 = pd.Series([36.0, 36.4, 37.0])
  r1 = [True, True, False]

  t2 = pd.Series([36.0, 37.0, np.nan])
  t3 = pd.Series([np.nan, np.nan, 35.0])
  t4 = pd.Series([np.nan, np.nan, np.nan])
  r2 = [False, False, False]

  assert r1 == median_test(t1)
  assert r2 == median_test(t2)
  assert r2 == median_test(t3)
  assert r2 == median_test(t4)
test_median_test()

def get_pass_median_test(plate_df, groupby_list):
  # make list that will become new df
  plate_df_with_median_test = []

  # iterate thru the dataframe, grouped by Sample
  # this gives us a mini-df with just one sample in each iteration
  for groupby_list, df in plate_df.groupby(groupby_list,  as_index=False):
    d = df.copy() # avoid set with copy warning

    # make new column 'median_test' that includes the results of the test
    d.loc[:, 'median_test'] = median_test(d.Cq)
    plate_df_with_median_test.append(d)

  # put the dataframe back together
  plate_df_with_median_test = pd.concat(plate_df_with_median_test)
  return(plate_df_with_median_test)

def get_pass_grubbs_test(plate_df, groupby_list):
  # make list that will become new df
  plate_df_with_grubbs_test = pd.DataFrame()

  # iterate thru the dataframe, grouped by Sample
  # this gives us a mini-df with just one sample in each iteration
  for groupby_list, df in plate_df.groupby(groupby_list,  as_index=False):
    d = df.copy() # avoid set with copy warning
    # d.Cq=[round(n, 2) for n in d.Cq]
    # make new column 'grubbs_test' that includes the results of the test
    if (len(d.Cq.dropna())<3): #cannot evaluate for fewer than 3 values

        if (len(d.Cq.dropna())==2) & (np.std(d.Cq.dropna()) <0.2): #got this from https://www.gene-quantification.de/dhaene-hellemans-qc-data-2010.pdf
            d.loc[:, 'grubbs_test'] = True
            plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)
        else:
            d.loc[:, 'grubbs_test'] = False
            plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)

    else:

        b=list(d.Cq) #needs to be given unindexed list
        outliers=grubbs.max_test_outliers(b, alpha=0.025)
        if len(outliers) > 0:
            d.loc[:, 'grubbs_test'] = True
            d.loc[d.Cq.isin(outliers), 'grubbs_test'] = False
            plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)
        else:
            d.loc[:, 'grubbs_test'] = True
            plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)

  return(plate_df_with_grubbs_test)
  # put the dataframe back together
  plate_df_with_grubbs_test = pd.concat(plate_df_with_grubbs_test)
  return(plate_df_with_grubbs_test)

def compute_linear_info(plate_data):
    '''compute the information for linear regression

    Params
        plate_data: pandas dataframe with columns
            Cq_mean (already processed to remove outliers)
            log_Quantity
    Returns
        slope, intercept, r2, efficiency
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
    Flag outliers via Dixon's Q, homemade "median_test", ans/or grubbs test
    Calculate the Cq means, Cq stds, counts before & after removing outliers

    Params
    plate_df_in:
        qpcr data in pandas df, must be 1 plate with 1 target
        should be in the format from QuantStudio3 with
        columns 'Target', 'Sample', 'Cq'
    checks_include:
        which way to check for outliers options are
        ('all', 'dixonsq_only', 'std_check_only','grubbs_only', None)
    Returns
    plate_df: same data, with additional columns depending on checks_include
        pass_dixonsq (0 or 1)
        median_test (True or False)
        Cq_mean (calculated mean of Cq after excluding outliers)
    '''

    if (checks_include not in ['all', 'dixonsq_only', 'median_only','grubbs_only', None]):
        raise ValueError('''invalid input, must be one of the following: 'all',
                      'dixonsq_only', 'median_only'or None''')

    if len(plate_df_in.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')

    plate_df = plate_df_in.copy() # fixes pandas warnings

    groupby_list = ['plate_id', 'Sample', 'Sample_plate',
                    'Target','Task', 'inhibition_testing']

    # make copy of Cq column and later turn this to np.nan for outliers
    plate_df['Cq_copy'] = plate_df['Cq'].copy()

    # Test triplicates with Dixon's Q
    #use_dixonsq = False
    if checks_include in ['all', 'dixonsq_only']:
        #use_dixonsq = True
        plate_df = get_pass_dixonsq(plate_df, ['Sample'])
        # convert point failing the outlier test(s) to np.nan in Cq_copy column
        plate_df.loc[plate_df.pass_dixonsq==0, 'Cq_copy'] = np.nan

    # Test triplicates with another outlier test? (same format as for dixonsq)
    if checks_include in ['all', 'median_only']:
       plate_df = get_pass_median_test(plate_df, ['Sample'])
       plate_df.loc[plate_df.median_test == False, 'Cq_copy'] = np.nan

    # Test triplicates with another outlier test? (same format as for dixonsq)
    if checks_include in ['all', 'grubbs_only']:
       plate_df = get_pass_grubbs_test(plate_df, ['Sample'])
       plate_df.loc[plate_df.grubbs_test == False, 'Cq_copy'] = np.nan

    # summarize to get mean, std, counts with and without outliers removed

    plate_df_avg = plate_df.groupby(groupby_list).agg(
                                               template_volume=('template_volume','max'),
                                               Q_init_mean=('Quantity', lambda x: np.nan if all(np.isnan(x)) else sci.gmean(x.dropna(),axis=0)),
                                               Q_init_std=('Quantity', lambda x: np.nan if ( (len(x.dropna()) <2 )| all(np.isnan(x)) ) else (sci.gstd(x.dropna(),axis=0))),
                                               Q_init_CoV=('Quantity',lambda x: np.std(x.dropna()) / np.mean(x.dropna())),
                                               Cq_init_mean=('Cq', 'mean'),
                                               Cq_init_std=('Cq', 'std'),
                                               Cq_init_min=('Cq', 'min'),
                                               replicate_init_count=('Cq','count'),
                                               Cq_mean=('Cq', lambda x:  np.nan if all(np.isnan(x)) else sci.gmean(x.dropna(),axis=0)),
                                               Cq_std=('Cq', lambda x: np.nan if ((len(x.dropna()) <2 )| all(np.isnan(x)) ) else (sci.gstd(x.dropna(),axis=0))),
                                               replicate_count=('Cq_copy', 'count'),
                                               is_undetermined_count=('is_undetermined', 'sum')
                                               )
    # note: count in agg will exclude nan
    plate_df_avg = plate_df_avg.reset_index()

    return(plate_df, plate_df_avg)

def process_standard(plate_df):
    '''
    from single plate with single target, calculate standard curve

    Params:
        plate_df: output from combine_triplicates(); df containing Cq_mean
        must be single plate with single target
    Returns
        num_points: number of points used in new std curve
        Cq_of_lowest_std_quantity: the Cq value of the lowest pt used in the new std curve
        lowest_std_quantity: the Quantity value of the lowest pt used in the new std curve
        slope:
        intercept:
        r2:
        efficiency:
    '''
    if len(plate_df.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')
    #what is the lowest sample Cq and quantity on this plate
    standard_df = plate_df[plate_df.Task == 'Standard'].copy()

    # require at least 2 triplicates or else convert to nan
    standard_df = standard_df[standard_df.replicate_count > 1]

    standard_df['log_Quantity'] = standard_df.apply(lambda row: np.log10(pd.to_numeric(row.Q_init_mean)), axis = 1)
    std_curve_df = standard_df[['Cq_mean', 'log_Quantity']].drop_duplicates().dropna()
    num_points = std_curve_df.shape[0]
    Cq_of_lowest_std_quantity = max(standard_df.Cq_mean)
    sort_a=standard_df.sort_values(by='Cq_mean',ascending=True).copy()
    Cq_of_2ndlowest_std_quantity = sort_a.Cq_mean[1]
    lowest_std_quantity = np.nan
    sort_b=standard_df.sort_values(by='log_Quantity',ascending=True).copy()
    print(sort_b.log_Quantity)
    lowest_std_quantity2nd= 10**(sort_b.log_Quantity[1])
    print(lowest_std_quantity2nd)
    slope, intercept, r2, efficiency = (np.nan, np.nan, np.nan, np.nan)

    if num_points > 2:
        lowest_std_quantity = 10**min(standard_df.log_Quantity)
        slope, intercept, r2, efficiency = compute_linear_info(std_curve_df)

    return(num_points, Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity, lowest_std_quantity, lowest_std_quantity2nd, slope, intercept, r2, efficiency)

def process_unknown(plate_df, std_curve_info):
    '''
    Calculates quantity based on Cq_mean and standard curve
    Params
        plate_df: output from combine_triplicates(); df containing Cq_mean
        must be single plate with single target
        std_curve_info: output from process_standard() as a list
    Returns
        unknown_df: the unknown subset of plate_df, with 2 new columns
        (Quantity_mean, and q_diff)
        Cq_of_lowest_sample_quantity: the Cq value of the lowest pt used on the plate
        these columns represent the recalculated quantity using Cq mean and the
        slope and intercept from the std curve
    '''

    [num_points, Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity, lowest_std_quantity, lowest_std_quantity2nd, slope, intercept, r2, efficiency] = std_curve_info
    unknown_df = plate_df[plate_df.Task == 'Unknown'].copy()
    unknown_df['Cq_of_lowest_sample_quantity']=np.nan
    if len(unknown_df.Task)==0:
        unknown_df['Cq_of_lowest_sample_quantity']=np.nan
    else:
        if all(np.isnan(unknown_df.Cq_mean)):
            unknown_df['Cq_of_lowest_sample_quantity']= np.nan #avoid error
        else:
            unknown_df['Cq_of_lowest_sample_quantity']=np.nanmax(unknown_df.Cq_mean)
    unknown_df['Quantity_mean'] = np.nan
    unknown_df['Quantity_mean_upper_std'] = np.nan
    unknown_df['Quantity_mean_lower_std'] = np.nan
    unknown_df['q_diff'] = np.nan
    unknown_df['Quantity_mean'] = 10**((unknown_df['Cq_mean'] - intercept)/slope)
    unknown_df['Quantity_mean_lower_std'] = 10**((unknown_df['Cq_mean'] + unknown_df['Cq_std'] - intercept)/slope)
    unknown_df['Quantity_mean_upper_std'] = 10**((unknown_df['Cq_mean'] - unknown_df['Cq_std'] - intercept)/slope)
    unknown_df.loc[unknown_df[unknown_df.Cq_mean == 0].index, 'Quantity_mean'] = np.nan
    unknown_df['q_diff'] = unknown_df['Q_init_mean'] - unknown_df['Quantity_mean']
    return(unknown_df)

def process_ntc(plate_df):
    ntc = plate_df[plate_df.Task == 'Negative Control']
    ntc_result = np.nan
    if ntc.is_undetermined.all():
        ntc_result = 'negative'
    else:
        if all(np.isnan(ntc.Cq)):
            ntc_result= np.nan #avoid error
        else:
            ntc_result = np.nanmin(ntc.Cq)
    return(ntc_result)



def determine_samples_BLoD(raw_outliers_flagged_df, cutoff, checks_include):
        '''
        For each target in raw qpcr data, this function defines the limit of quantification as the fraction of qpcr replicates at a quantity that are detectable
        It works depending on which test was selected, so if grubbs was selected, it only evaluates for replicates that pass grubbs

        Params:
            Task
            Quantity
            Target
            Cq
            Sample
        Returns
            a dataframe with Target and the limit of detection
        '''
        dfm= raw_outliers_flagged_df
        if checks_include in ['all', 'grubbs_only']:
            dfm=dfm[dfm.grubbs_test==True].copy()
        if checks_include in ['all', 'median_only']:
            dfm=dfm[dfm.median_test==True].copy()
        if checks_include in ['all', 'dixonsq_only']:
            dfm=dfm[dfm.pass_dixonsq==True].copy()

        dfm=dfm[dfm.Task=='Standard'] #only standards
        dfm=dfm[dfm.Quantity!=0] #no NTCs
        out_fin=pd.DataFrame(columns=["Target","LoD_Cq","LoD_Quantity"]) #empty dataframe with desired columns

        #iterate through targets, groupby quantity, and determine the fraction of the replicates that were detectable
        targs=dfm.Target.unique()
        for target in targs:
            df_t=dfm[dfm.Target==target].copy()
            out=df_t.groupby(["Quantity"]).agg(
                                    Cq_mean=('Cq', lambda x:  np.nan if all(np.isnan(x)) else sci.gmean(x.dropna(),axis=0)),
                                    positives=('Cq','count'),
                                    total=('Sample', 'count')).reset_index()
            out['fr_pos']=out.positives/out.total

            #only take the portion of the dataframe that is greater than the cutoff
            out=out[out.fr_pos > cutoff ].copy()
            #something is there hopefully but if not
            if len(out.fr_pos)<1:
                out_fin=out_fin.append(pd.DataFrame({'Target':target, "LoD_Cq": np.nan, "LoD_Quantity":np.nan}), ignore_index=True)

            #usual case for N1/ bCov
            fin=out[out.fr_pos==min(out.fr_pos)].copy()
            if len(fin.fr_pos) ==1:
                out_fin=out_fin.append(pd.DataFrame({'Target':target, "LoD_Cq": fin.Cq_mean, "LoD_Quantity":fin.Quantity}), ignore_index=True)
            #usual case for PMMoV/18S
            elif len(fin.fr_pos)>1:
                fin=out[(out.fr_pos==min(out.fr_pos))&(out.Quantity==min(out.Quantity))].copy()
                out_fin=out_fin.append(pd.DataFrame({'Target':target, "LoD_Cq": fin.Cq_mean, "LoD_Quantity":fin.Quantity}), ignore_index=True)
        return (out_fin)


def determine_samples_BLoQ(qpcr_p, max_cycles, out_fin, include_LoD=False):
    '''
    from processed unknown qpcr data and the max cycles allowed (usually 40) this will return qpcr_processed with a boolean column indicating samples bloq.
    samples that have Cq_mean that is nan are classified as bloq (including true negatives and  samples removed during processing)
    If include LoD is true, out_fin comes from determines_samples_BLO

    Params:
        Cq_mean the combined triplicates of the sample
        Cq_of_lowest_sample_quantity the max cq of the samples on the plate

    Returns
        same data with column bloq a boolean column indicating if the sample is below the limit of quantification
    '''

    if include_LoD:
        qpcr_p["blod"]= np.nan
        targs=qpcr_p.Target.unique()
        for target in targs:
            C_value=float(out_fin.loc[(out_fin.Target==target),"LoD_Cq"])
            Q_value=float(out_fin.loc[(out_fin.Target==target),"LoD_Quantity"])
            if np.isnan(C_value):
                qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.Cq_mean > C_value),"bloq"]= np.nan
            else:
                qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.Cq_mean > C_value),"bloq"]= True

                qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.Cq_of_lowest_std_quantity> C_value),"Cq_of_lowest_std_quantity"]= qpcr_p.Cq_of_2ndlowest_std_quantity
                qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.Cq_of_lowest_std_quantity> C_value),"lowest_std_quantity"]= qpcr_p.lowest_std_quantity2nd

    qpcr_p['bloq']=np.nan
    qpcr_p.loc[(np.isnan(qpcr_p.Cq_mean)),'bloq']= True
    qpcr_p.loc[(qpcr_p.Cq_mean >= max_cycles),'bloq']= True
    qpcr_p.loc[(qpcr_p.Cq_mean > qpcr_p.Cq_of_lowest_std_quantity),'bloq']= True
    qpcr_p.loc[(qpcr_p.Cq_mean <= qpcr_p.Cq_of_lowest_std_quantity)&(qpcr_p.Cq_mean < max_cycles),'bloq']= False



    return(qpcr_p)

def process_qpcr_raw(qpcr_raw, checks_include,include_LoD=False,cutoff=0.9):
    '''wrapper to process whole sheet at once by plate_id and Target
    params
    qpcr_raw: df from read_qpcr_data()
    checks_include: how to remove outliers ('all', 'dixonsq_only', 'median_only')
    optional:
    include_LoD adds blod column and moves lowest standard quantity and Cq of lowest standard quantity based on LoD
    cutoff is the fraction of positive replicates in standard curves allowed to consider that standard curve point detectable (used if previous is true)
    '''

    if (checks_include not in ['all', 'dixonsq_only', 'median_only','grubbs_only', None]):
        raise ValueError('''invalid input, must be one of the following: 'all',
                      'dixonsq_only', 'median_only'or None''')
    std_curve_df = []
    qpcr_processed = []
    raw_outliers_flagged_df = []
    for [plate_id, target], df in qpcr_raw.groupby(["plate_id", "Target"]):

        ntc_result = process_ntc(df)
        outliers_flagged, no_outliers_df = combine_triplicates(df, checks_include)

        # define outputs and fill with default values
        num_points, Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity,lowest_std_quantity,lowest_std_quantity2nd, slope, intercept, r2, efficiency = np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        unknown_df = df[df.Task == 'Unknown']

        # if there are >3 pts in std curve, calculate stats and recalculate quants
        num_points = no_outliers_df[no_outliers_df.Task == 'Standard'].drop_duplicates('Sample').shape[0]
        if num_points > 3:
            num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity, lowest_std_quantity, lowest_std_quantity2nd,slope, intercept, r2, efficiency = process_standard(no_outliers_df)
            std_curve_info = [num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity,lowest_std_quantity, lowest_std_quantity2nd,slope, intercept, r2, efficiency]
            unknown_df = process_unknown(no_outliers_df, std_curve_info)
        std_curve_df.append([plate_id, target, num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity,lowest_std_quantity,lowest_std_quantity2nd, slope, intercept, r2, efficiency, ntc_result])
        qpcr_processed.append(unknown_df)
        raw_outliers_flagged_df.append(outliers_flagged)

    # compile into dataframes
    raw_outliers_flagged_df = pd.concat(raw_outliers_flagged_df)
    out_fin=determine_samples_BLoD(raw_outliers_flagged_df, cutoff, checks_include)
    std_curve_df = pd.DataFrame.from_records(std_curve_df,
                                             columns = ['plate_id',
                                                        'Target',
                                                        'num_points',
                                                        'Cq_of_lowest_std_quantity',
                                                        'Cq_of_2ndlowest_std_quantity',
                                                        'lowest_std_quantity',
                                                        'lowest_std_quantity2nd',
                                                        'slope',
                                                        'intercept',
                                                        'r2',
                                                        'efficiency',
                                                        'ntc_result'])
    qpcr_processed = pd.concat(qpcr_processed)
    qpcr_processed = qpcr_processed.merge(std_curve_df, how='left', on=['plate_id', 'Target'])
    qpcr_m=qpcr_processed[["plate_id","Cq_of_lowest_sample_quantity"]].copy().drop_duplicates(keep='first')
    std_curve_df=std_curve_df.merge(qpcr_m, how='left')
    qpcr_processed= determine_samples_BLoQ(qpcr_processed, 40, out_fin, cutoff)
    return(qpcr_processed, std_curve_df, raw_outliers_flagged_df)
