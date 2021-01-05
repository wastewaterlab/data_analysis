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
import scikit_posthocs as sp

import warnings

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


def get_pass_median_test(plate_df, groupby_list):
    #This is actually old grubbs test using outlier utils and 1-sided max
    # make list that will become new df
    plate_df_with_oldgrubbs_test = pd.DataFrame()

    # iterate thru the dataframe, grouped by Sample
    # this gives us a mini-df with just one sample in each iteration
    for groupby_list, df in plate_df.groupby(groupby_list,  as_index=False):
      d = df.copy() # avoid set with copy warning
      # d.Cq=[round(n, 2) for n in d.Cq]
      # make new column 'grubbs_test' that includes the results of the test
      if (len(d.Cq.dropna())<3): #cannot evaluate for fewer than 3 values

          if (len(d.Cq.dropna())==2) & (np.std(d.Cq.dropna()) <0.2): #got this from https://www.gene-quantification.de/dhaene-hellemans-qc-data-2010.pdf
              d.loc[:, 'median_test'] = True
              plate_df_with_oldgrubbs_test=plate_df_with_oldgrubbs_test.append(d)
          else:
              d.loc[:, 'median_test'] = False
              plate_df_with_oldgrubbs_test=plate_df_with_oldgrubbs_test.append(d)

      else:

          b=list(d.Cq) #needs to be given unindexed list
          outliers=grubbs.max_test_outliers(b, alpha=0.05)
          if len(outliers) > 0:
              d.loc[:, 'median_test'] = True
              d.loc[d.Cq.isin(outliers), 'median_test'] = False
              plate_df_with_oldgrubbs_test=plate_df_with_oldgrubbs_test.append(d)
          else:
              d.loc[:, 'median_test'] = True
              plate_df_with_oldgrubbs_test=plate_df_with_oldgrubbs_test.append(d)

    return(plate_df_with_oldgrubbs_test)
    # put the dataframe back together
    # plate_df_with_grubbs_test = pd.concat(plate_df_with_grubbs_test)
    # return(plate_df_with_grubbs_test)

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
        # outliers=grubbs.max_test_outliers(b, alpha=0.025)
        nonoutliers= sp.outliers_grubbs(b)
        outlier_len=len(b)-len(nonoutliers)
        if outlier_len > 0:
            d.loc[:, 'grubbs_test'] = False
            d.loc[d.Cq.isin(nonoutliers), 'grubbs_test'] = True
            plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)
        else:
            d.loc[:, 'grubbs_test'] = True
            plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)
  return(plate_df_with_grubbs_test)
  # put the dataframe back together
  # plate_df_with_grubbs_test = pd.concat(plate_df_with_grubbs_test)
  # return(plate_df_with_grubbs_test)

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
        Q_QuantStudio_std (calculated standard deviation based on QuantStudio output) for intrassay coefficient of variation
    '''

    if (checks_include not in ['all', 'dixonsq_only', 'median_only','grubbs_only', None]):
        raise ValueError('''invalid input, must be one of the following: 'all', 'grubbs_only',
                      'dixonsq_only', 'median_only'or None''')

    if len(plate_df_in.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')

    plate_df = plate_df_in.copy() # fixes pandas warnings

    groupby_list = ['plate_id', 'Sample', 'sample_full','Sample_plate',
                    'Target','Task', 'inhibition_testing','is_dilution',"dilution"]

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
                                               raw_Cq_values=('Cq',list),
                                               template_volume=('template_volume','max'),
                                               Q_init_mean=('Quantity','mean'), #only needed to preserve quantity information for standards later
                                               Q_init_std=('Quantity','std'),
                                               Q_init_gstd=('Quantity', lambda x: np.nan if ( (len(x.dropna()) <2 )| all(np.isnan(x)) ) else (sci.gstd(x.dropna(),axis=0))),
                                               # Q_QuantStudio_std = ('Quantity', 'std'),
                                               Cq_init_mean=('Cq', 'mean'),
                                               Cq_init_std=('Cq', 'std'),
                                               Cq_init_min=('Cq', 'min'),
                                               replicate_init_count=('Cq','count'),
                                               Cq_mean=('Cq_copy', 'mean'),
                                               Cq_std=('Cq_copy', 'std'),
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
        Cq_of_lowest_std_quantity_gsd: geometric standard dviation of the Cq of the lowest standard quantity
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

    standard_df['log_Quantity'] = np.log10(standard_df['Q_init_mean'])
    std_curve_df = standard_df[['Cq_mean', 'log_Quantity', "Cq_std"]].drop_duplicates().dropna()
    num_points = std_curve_df.shape[0]

    if (all(standard_df.Cq_mean == "") | len(standard_df.Cq_mean) <2):
        slope, intercept, r2, efficiency,Cq_of_lowest_std_quantity,Cq_of_2ndlowest_std_quantity,Cq_of_lowest_std_quantity_gsd,Cq_of_2ndlowest_std_quantity_gsd,lowest_std_quantity,lowest_std_quantity2nd = np.nan,np.nan,np.nan, np.nan, np.nan, np.nan,np.nan, np.nan,np.nan, np.nan
    else:
        #find the Cq of the lowest and second lowest (for LoQ) standard quantity
        Cq_of_lowest_std_quantity = max(standard_df.Cq_mean)
        sort_a=standard_df.sort_values(by='Cq_mean',ascending=True).copy().reset_index()
        Cq_of_2ndlowest_std_quantity = sort_a.Cq_mean[1]

        #find the geometric standard deviation of the Cq of the lowest and second lowest (for LoQ) standard quantity
        sort_a=standard_df.sort_values(by='Cq_mean',ascending=True).copy().reset_index()
        Cq_of_lowest_std_quantity_gsd = sort_a.Cq_std[0]
        Cq_of_2ndlowest_std_quantity_gsd = sort_a.Cq_std[1]

        # the  lowest and second lowest (for LoQ) standard quantity
        lowest_std_quantity = np.nan
        sort_b=standard_df.sort_values(by='Q_init_mean',ascending=True).copy().reset_index()
                # the lowest and second lowest (for LoQ) standard quantity
        lowest_std_quantity2nd = sort_b.Q_init_mean.values[1]
        slope, intercept, r2, efficiency = (np.nan, np.nan, np.nan, np.nan)

        if num_points > 2:
            lowest_std_quantity = sort_b.Q_init_mean.values[0]
            slope, intercept, r2, efficiency = compute_linear_info(std_curve_df)

    return(num_points, Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity, lowest_std_quantity, lowest_std_quantity2nd,Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd, slope, intercept, r2, efficiency)

def process_unknown(plate_df, std_curve_info):
    '''
    Calculates quantity based on Cq_mean and standard curve
    Params
        plate_df: output from combine_triplicates(); df containing Cq_mean
        must be single plate with single target
        std_curve_info: output from process_standard() as a list
    Returns
        unknown_df: the unknown subset of plate_df, with new columns
        Quantity_mean
        q_diff
        Cq_of_lowest_sample_quantity: the Cq value of the lowest pt used on the plate
        these columns represent the recalculated quantity using Cq mean and the
        slope and intercept from the std curve
        qpcr_coefficient_var the coefficient of variation for qpcr technical triplicates
        intraassay_var intraassay variation (arithmetic mean of the coefficient of variation for all triplicates on a plate)
    '''

    [num_points, Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity, lowest_std_quantity, lowest_std_quantity2nd,Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd, slope, intercept, r2, efficiency] = std_curve_info
    unknown_df = plate_df[plate_df.Task != 'Standard'].copy()
    unknown_df['Cq_of_lowest_sample_quantity'] = np.nan
    unknown_df['percent_CV']=(unknown_df['Q_init_std']-1)*100#the geometric std - 1 is the coefficient of variation using quant studio quantities to capture all the variation in the plate
    if all(np.isnan(unknown_df['percent_CV'])):
        unknown_df['intraassay_var'] = np.nan #avoid error
    else:
        unknown_df['intraassay_var']= np.nanmean(unknown_df['percent_CV'])

    # Set the Cq of the lowest std quantity for different ssituations
    if len(unknown_df.Task) == 0: #only standard curve plate
        unknown_df['Cq_of_lowest_sample_quantity'] = np.nan
    else:
        if all(np.isnan(unknown_df.Cq_mean)): #plate with all undetermined samples
            unknown_df['Cq_of_lowest_sample_quantity']= np.nan #avoid error
        else:
            targs=unknown_df.Target.unique() #other  plates (most  cases)
            for target in targs:
                unknown_df.loc[(unknown_df.Target==target),'Cq_of_lowest_sample_quantity']=np.nanmax(unknown_df.loc[(unknown_df.Target==target),'Cq_mean']) #because of xeno

    unknown_df['Quantity_mean'] = np.nan
    unknown_df['q_diff'] = np.nan
    unknown_df['Quantity_mean'] = 10**((unknown_df['Cq_mean'] - intercept)/slope)

    #initialize columns
    unknown_df['Quantity_std_combined_after']=np.nan
    unknown_df['Quantity_mean_combined_after']=np.nan
    for row in unknown_df.itertuples():
        ix=row.Index
        filtered_1= [element for element in row.raw_Cq_values if ~np.isnan(element) ] #initial nas
        filtered= [10**((element - intercept)/slope) for element in filtered_1]
        if(len(filtered)>1):
                filtered= [element for element in filtered if ~np.isnan(element) ] #nas introduced when slope and interceptna
                if(len(filtered)>1):
                    if (row.Target != "Xeno"):
                        unknown_df.loc[ix,"Quantity_mean_combined_after"]=sci.gmean(filtered)
                        if(all(x >0 for x in filtered)):
                            unknown_df.loc[ix,"Quantity_std_combined_after"]=sci.gstd(filtered)


    # if Cq_mean is zero, don't calculate a quantity (turn to NaN)
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
            ntc_result = np.nan #avoid error
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
        assay_assessment_df=pd.DataFrame(columns=["Target","LoD_Cq","LoD_Quantity"]) #empty dataframe with desired columns

        #iterate through targets, groupby quantity, and determine the fraction of the replicates that were detectable
        targs=dfm.Target.unique()
        for target in targs:
            print(target)
            df_t=dfm[dfm.Target==target].copy()
            out=df_t.groupby(["Quantity"]).agg(
                                    Cq_mean=('Cq', lambda x:  np.nan if all(np.isnan(x)) else sci.gmean(x.dropna(),axis=0)),
                                    positives=('Cq','count'),
                                    total=('Sample', 'count')).reset_index()
            out['fr_pos']=out.positives/out.total

            #only take the portion of the dataframe that is greater than the cutoff
            out=out[out.fr_pos > cutoff ].copy()
            fin=np.nan
            print(out)
            #something is there hopefully but if not
            if len(out.fr_pos)<1:
                assay_assessment_df=assay_assessment_df.append(pd.DataFrame({'Target':target, "LoD_Cq": np.nan, "LoD_Quantity":np.nan}), ignore_index=True)
            elif len(out.fr_pos) ==1:
                assay_assessment_df=assay_assessment_df.append(pd.DataFrame({'Target':target, "LoD_Cq": out.Cq_mean, "LoD_Quantity":out.Quantity}), ignore_index=True)
            elif len(out.fr_pos)>1:
                if out.loc[out.Quantity==min(out.Quantity),"fr_pos"].item()==1:
                    fin=out.loc[out.Quantity==min(out.Quantity),:].copy()
                    assay_assessment_df=assay_assessment_df.append(pd.DataFrame({'Target':target, "LoD_Cq": fin.Cq_mean, "LoD_Quantity":fin.Quantity}), ignore_index=True)
                else:
                    fin=out[(out.fr_pos==min(out.fr_pos))&(out.Quantity==min(out.Quantity))].copy()
                    assay_assessment_df=assay_assessment_df.append(pd.DataFrame({'Target':target, "LoD_Cq": fin.Cq_mean, "LoD_Quantity":fin.Quantity}), ignore_index=True)
        print(assay_assessment_df)
        return (assay_assessment_df)


def determine_samples_BLoQ(qpcr_p, max_cycles, assay_assessment_df, include_LoD=False):
    '''
    from processed unknown qpcr data and the max cycles allowed (usually 40) this will return qpcr_processed with a boolean column indicating samples bloq.
    samples that have Cq_mean that is nan are classified as bloq (including true negatives and  samples removed during processing)
    If include LoD is true, assay_assessment_df comes from determines_samples_BLO

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
            if (len(assay_assessment_df.loc[(assay_assessment_df.Target==target),"LoD_Cq"])>0):
                C_value=float(assay_assessment_df.loc[(assay_assessment_df.Target==target),"LoD_Cq"])
                Q_value=float(assay_assessment_df.loc[(assay_assessment_df.Target==target),"LoD_Quantity"])
                if np.isnan(C_value):
                    qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.Cq_mean > C_value),"blod"]= np.nan
                else:
                    qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.Cq_mean > C_value),"blod"]= True
                    qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.Cq_mean <= C_value),"blod"]= False
                    qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.blod==True),"Cq_of_lowest_std_quantity"]= qpcr_p.Cq_of_2ndlowest_std_quantity
                    qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.blod==True),"Cq_of_lowest_std_quantity_gsd"]= qpcr_p.Cq_of_2ndlowest_std_quantity_gsd
                    qpcr_p.loc[(qpcr_p.Target==target)&(qpcr_p.blod==True),"lowest_std_quantity"]= qpcr_p.lowest_std_quantity2nd

    qpcr_p['bloq']=np.nan
    qpcr_p.loc[(np.isnan(qpcr_p.Cq_mean)),'bloq']= True
    qpcr_p.loc[(qpcr_p.Cq_mean >= max_cycles),'bloq']= True
    qpcr_p.loc[(qpcr_p.Cq_mean > qpcr_p.Cq_of_lowest_std_quantity),'bloq']= True
    qpcr_p.loc[(qpcr_p.Cq_mean <= qpcr_p.Cq_of_lowest_std_quantity)&(qpcr_p.Cq_mean < max_cycles),'bloq']= False



    return(qpcr_p)

def process_dilutions(qpcr_p):
    '''
    from processed unknown qpcr data this function:
    1. looks for diluted samples
    2. adjusts quantities for the dilution,
    4. Makes these into a new dilution_experiments dataframe
    5. checks that there are no other non dilution combinations of sample and target for the samples marked as dilutions
        in the origional dataframe. If there are, it removes the diluted sample(s) from the dataframe and throws a warning with the sample name
    6. checks if  there are multiple entries with the same sample name and  target in the diluted samples:
        if there are it chooses the one with the highest N1 value and keeps that in the dataframe and removes the other one (both will be in dilution_experiments)

    Params:
        is_dilution
        Target
        Quantity_mean
        dilution
    Returns
        same dataframe without duplicated process_dilutions
        a new dataframe with all dilutions
    '''
    dilution_expts_df=qpcr_p
    remove=list()

    if(len(qpcr_p.loc[(qpcr_p.is_dilution=='Y')]) > 0):
        qpcr_p.loc[(qpcr_p.is_dilution=='Y')&(np.isnan(qpcr_p.Quantity_mean)), "Quantity_mean"]=0
        qpcr_p.loc[(qpcr_p.is_dilution=='Y'), "Quantity_mean"]= qpcr_p.loc[(qpcr_p.is_dilution=='Y'), "Quantity_mean"] * qpcr_p.loc[(qpcr_p.is_dilution=='Y'), "dilution"]
        dilution_expts_df=qpcr_p.loc[(qpcr_p.is_dilution=='Y'), ].copy()
        check=dilution_expts_df.groupby(["Sample", "Target"])["dilution"].count().reset_index()

        all_samps= qpcr_p.loc[(qpcr_p.is_dilution!='Y'), ].copy()
        all_samps["warnid"]=all_samps.Sample.str.cat(all_samps.Target, sep ="_")
        all_samps=all_samps["warnid"].unique()

        for row in check.itertuples():
            targ=row.Target
            samp=row.Sample
            wid= "_".join([targ,samp])
            if row.dilution >1:
                all_idx=qpcr_p.loc[(qpcr_p.is_dilution=='Y')&(qpcr_p.Sample==samp)&(qpcr_p.Target==targ),"Quantity_mean"].index.values.tolist()
                max_idx=qpcr_p.loc[(qpcr_p.is_dilution=='Y')&(qpcr_p.Sample==samp)&(qpcr_p.Target==targ),"Quantity_mean"].idxmax()
                lis=list([x for x in all_idx if x != max_idx])
                remove = remove + lis
                if wid in all_samps:
                    warnings.warn("\n\n\n{} is double listed as a dilution sample and a non dilution sample. Change one is_primary_value. Currently the dilution value is removed in the code.\n\n\n".format(samp))
                    remove.append(max_idx)


            else:
                if wid in all_samps:
                    warnings.warn("\n\n\n{} is double listed as a dilution sample and a non dilution sample. Change one is_primary_value. Currently the dilution value is removed in the code.\n\n\n".format(samp))
                    idx=qpcr_p.loc[(qpcr_p.is_dilution=='Y')&(qpcr_p.Sample==samp)&(qpcr_p.Target==targ),"Quantity_mean"].index.values.tolist()
                    remove.append(idx)


    if not remove:
        qpcr_p=qpcr_p
    elif len(remove)==1:
        remove=remove[0]
        qpcr_p=qpcr_p.drop(remove)
    else:
        qpcr_p=qpcr_p.loc[~qpcr_p.index.isin(remove)].copy()

    return(qpcr_p,dilution_expts_df)

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
        raise ValueError('''invalid input, must be one of the following: 'all', 'grubs_only',
                      'dixonsq_only', 'median_only'or None''')
    std_curve_df = []
    qpcr_processed = []
    raw_outliers_flagged_df = []
    for [plate_id, target], df in qpcr_raw.groupby(["plate_id", "Target"]):

        ntc_result = process_ntc(df)
        outliers_flagged, no_outliers_df = combine_triplicates(df, checks_include)

        # define outputs and fill with default values
        num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity,lowest_std_quantity,lowest_std_quantity2nd, Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd,slope, intercept, r2, efficiency = np.nan, np.nan,np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # if there are >3 pts in std curve, calculate stats and recalculate quants
        num_points = no_outliers_df[no_outliers_df.Task == 'Standard'].drop_duplicates('Sample').shape[0]
        num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity, lowest_std_quantity, lowest_std_quantity2nd,Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd, slope, intercept, r2, efficiency = process_standard(no_outliers_df)
        std_curve_info = [num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity,lowest_std_quantity, lowest_std_quantity2nd,Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd,slope, intercept, r2, efficiency]
        unknown_df = process_unknown(no_outliers_df, std_curve_info)
        std_curve_df.append([plate_id, target, num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity,lowest_std_quantity,lowest_std_quantity2nd, Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd, slope, intercept, r2, efficiency, ntc_result])
        qpcr_processed.append(unknown_df)
        raw_outliers_flagged_df.append(outliers_flagged)

    # compile into dataframes
    raw_outliers_flagged_df = pd.concat(raw_outliers_flagged_df)
    assay_assessment_df=determine_samples_BLoD(raw_outliers_flagged_df, cutoff, checks_include)
    std_curve_df = pd.DataFrame.from_records(std_curve_df,
                                             columns = ['plate_id',
                                                        'Target',
                                                        'num_points',
                                                        'Cq_of_lowest_std_quantity',
                                                        'Cq_of_2ndlowest_std_quantity',
                                                        'lowest_std_quantity',
                                                        'lowest_std_quantity2nd',
                                                        'Cq_of_lowest_std_quantity_gsd',
                                                        'Cq_of_2ndlowest_std_quantity_gsd',
                                                        'slope',
                                                        'intercept',
                                                        'r2',
                                                        'efficiency',
                                                        'ntc_result'])
    qpcr_processed = pd.concat(qpcr_processed)
    qpcr_processed = qpcr_processed.merge(std_curve_df, how='left', on=['plate_id', 'Target'])
    qpcr_processed,dilution_expts_df = process_dilutions(qpcr_processed)
    control_df=qpcr_processed[(qpcr_processed.Sample.str.contains("control"))|(qpcr_processed.Task!="Unknown")].copy()
    qpcr_processed=qpcr_processed[qpcr_processed.Task=="Unknown"].copy()

    #make  columns calculated in other functions to go in the standard curve info
    qpcr_m=qpcr_processed[["plate_id","Target","Cq_of_lowest_sample_quantity",'intraassay_var']].copy().drop_duplicates(keep='first')
    std_curve_df=std_curve_df.merge(qpcr_m, how='left') # add Cq_of_lowest_sample_quantity and intraassay variation

    qpcr_processed= determine_samples_BLoQ(qpcr_processed, 40, assay_assessment_df, cutoff)
    std_curve_df=std_curve_df.drop("Cq_of_2ndlowest_std_quantity", axis=1)
    std_curve_df=std_curve_df.drop("Cq_of_2ndlowest_std_quantity_gsd", axis=1)
    std_curve_df=std_curve_df.drop("lowest_std_quantity2nd", axis=1)
    std_curve_df=std_curve_df[std_curve_df.Target != "Xeno"].copy()

    #check for duplicates
    a=qpcr_processed[(qpcr_processed.Sample!="__")&(qpcr_processed.Sample!="")]
    a=a[a.duplicated(["Sample","Target"],keep=False)].copy()
    if len(a) > 0:
        plates=a.plate_id.unique()
        l=len(plates)
        warnings.warn("\n\n\n {0} plates have samples that are double listed in qPCR_Cts spreadsheet. Check the following plates and make sure one is_primary_value is set to N:\n\n\n{1}\n\n\n".format(l,plates))

    return(qpcr_processed, std_curve_df, dilution_expts_df,raw_outliers_flagged_df, control_df)
