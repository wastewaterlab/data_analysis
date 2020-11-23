import math
import numpy as np
import outliers
from outliers import smirnov_grubbs as grubbs
import pandas as pd
from pandas.api.types import CategoricalDtype
import pdb
from scipy import stats as sci
from scipy.stats import linregress
from scipy.stats.mstats import gmean
from scipy.stats import gstd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import resample
import sys
import warnings

## TODO: rewrite everything to handle sets of triplicates

def grubbs_test(replicates, max_std_for_2_reps=0.2, alpha=0.025):
    '''
    from list of triplicates, determine passing replicates

    Params
    replicates: list of Cq values for replicates (usually triplicate)
    max_std_for_2_reps: from https://www.gene-quantification.de/dhaene-hellemans-qc-data-2010.pdf
    alpha: alpha for grubb's test

    Returns
    list of replicates passing grubb's test
    '''

    replicates_out = []

    replicates_no_nan = [x for x in replicates if ~np.isnan(x)]
    if len(replicates_no_nan) >= 3:
        outliers = grubbs.max_test_outliers(replicates_no_nan, alpha)
        # drop the outliers from the list
        replicates_out = [x for x in replicates_no_nan if x not in outliers]
    elif (len(replicates_no_nan)) == 2 and (np.std(replicates_no_nan) < max_std_for_2_reps):
        replicates_out = replicates_no_nan

    return(replicates_out)

def get_gstd(replicates):
    gstd_value = np.nan
    replicates_no_nan = [x for x in replicates if ~np.isnan(x)]
    if len(replicates_no_nan) >= 2:
        gstd_value = sci.gstd(replicates_no_nan)
    return(gstd_value)


def get_gmean(replicates):
    gmean_value = np.nan
    replicates_no_nan = [x for x in replicates if ~np.isnan(x)]
    if len(replicates_no_nan) >= 1:
        gmean_value = sci.gmean(replicates_no_nan)
    return(gmean_value)


def combine_replicates(plate_df, collapse_on=['Sample', 'dilution', 'Target', 'Task']):
    '''
    collapse replicates with identical attributes as determined by collapse_on list
    calculate summary columns

    Params
    plate_df: preprocessed plate dataframe from read_qpcr_data()
        # remove Omit == 'TRUE'
        # remove inhibition testing plates (in case not cleaned)
        # remove blank Sample names (these should have been Omit == TRUE but sometimes not entered correctly)
        # create dilution column by splitting Sample
        # create is_undetermined column
    collapse_on: list of attributes that define technical replicates

    Returns
    plate_df: collapsed plate with summary columns for the combined technical replicates
    '''

    # collapse replicates
    plate_df = plate_df[['Sample', 'dilution', 'Target', 'Task', 'Cq', 'Quantity', 'is_undetermined']]
    plate_df = plate_df.groupby(collapse_on).agg(lambda x: x.tolist())
    plate_df = plate_df.reset_index()

    # create summary columns describing all replicates
    plate_df['Cq_no_outliers'] = plate_df.Cq.apply(lambda x: grubbs_test(x))

    plate_df['Cq_init_mean'] = plate_df.Cq.apply(np.mean)
    plate_df['Cq_init_std'] = plate_df.Cq.apply(lambda x: get_gstd(x))
    plate_df['Cq_init_min'] = plate_df.Cq.apply(np.min)
    plate_df['replicate_init_count'] = plate_df.Cq.apply(lambda x: len(x))

    plate_df['Q_init_mean'] = plate_df.Quantity.apply(np.mean)
    plate_df['Q_init_std'] = plate_df.Quantity.apply(lambda x: get_gstd(x))

    plate_df['Cq_mean'] = plate_df.Cq_no_outliers.apply(lambda x: get_gmean(x))
    plate_df['Cq_std'] = plate_df.Cq_no_outliers.apply(lambda x: get_gstd(x))
    plate_df['replicate_count'] = plate_df.Cq_no_outliers.apply(lambda x: len(x))
    plate_df['is_undetermined_count'] = plate_df.is_undetermined.apply(lambda x: sum(x))
    plate_df = plate_df.sort_values(['Target', 'Sample', 'dilution'])
    return(plate_df)


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


def process_standard(plate_df, loq_min_reps=(2/3)):
    '''
    from single plate with single target, calculate standard curve info

    Defines the limit of quantification as the lowest pt on the std curve
    where the fraction of detected replicates meets the cutoff
    Note: Evaluates only the replicates that pass grubb's test
    TODO decide if this is a fair way to evaluate LoD- Grubb's test seems too stringent on standards in particular

    Params:
        plate_df: output from combine_replicates(); df containing Cq_mean
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
        lod_Cq: the Cq for the limit of quantification
        lod_Quantity: the quantity for the limit of quantification
    '''
    if len(plate_df.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')

    # define outputs

    slope = np.nan
    intercept = np.nan
    r2 = np.nan
    efficiency = np.nan
    Cq_of_lowest_std_quantity = np.nan
    Cq_of_2ndlowest_std_quantity = np.nan
    Cq_of_lowest_std_quantity_gsd = np.nan
    Cq_of_2ndlowest_std_quantity_gsd = np.nan
    lowest_std_quantity = np.nan
    lowest_std_quantity2nd = np.nan
    loq_Cq = np.nan
    loq_Quantity = np.nan

    #what is the lowest sample Cq and quantity on this plate
    standard_df = plate_df[plate_df.Task == 'Standard'].copy()

    # require at least 2 replicates
    standard_df = standard_df[standard_df.replicate_count > 1]
    num_points = len(standard_df)

    if num_points >= 2:
        standard_df['log_Quantity'] = np.log10(standard_df['Q_init_mean'])

        #find the Cq_mean of the lowest and second lowest (for LoQ) standard quantity
        standard_df = standard_df.sort_values('Q_init_mean')
        Cq_of_lowest_std_quantity = standard_df.Cq_mean.values[0]
        Cq_of_2ndlowest_std_quantity = standard_df.Cq_mean.values[1]

        #find the geometric standard deviation of the Cq of the lowest and second lowest (for LoQ) standard quantity
        Cq_of_lowest_std_quantity_gsd = standard_df.Cq_std.values[0]
        Cq_of_2ndlowest_std_quantity_gsd = standard_df.Cq_std.values[1]

        # the lowest and second lowest (for LoQ) standard quantity
        lowest_std_quantity = standard_df.Q_init_mean.values[0]
        lowest_std_quantity2nd = standard_df.Q_init_mean.values[1]

        if num_points > 2:
            slope, intercept, r2, efficiency = compute_linear_info(standard_df)

    ## determine LoQ
    # determine the fraction of the replicates that were detectable
    standard_df['fraction_positive'] = np.nan
    standard_df['fraction_positive'] = standard_df.replicate_count / standard_df.replicate_init_count

    #only take the portion of the dataframe that is >= the cutoff
    standard_df = standard_df[standard_df.fraction_positive >= loq_min_reps].copy()

    # if there are standards that meet the cutoff
    # find the lowest quantity and report it and its Cq_mean as the LoD values
    if len(standard_df) > 0:
        standard_df = standard_df.sort_values('Q_init_mean')
        loq_Cq = standard_df.Cq_mean.values[0]
        loq_Quantity = standard_df.Q_init_mean.values[0]

    std_curve_info = (num_points,
                      Cq_of_lowest_std_quantity,
                      Cq_of_2ndlowest_std_quantity,
                      lowest_std_quantity,
                      lowest_std_quantity2nd,
                      Cq_of_lowest_std_quantity_gsd,
                      Cq_of_2ndlowest_std_quantity_gsd,
                      slope,
                      intercept,
                      r2,
                      efficiency,
                      loq_Cq,
                      loq_Quantity)

    return(std_curve_info)


def process_unknown(plate_df, std_curve_intercept, std_curve_slope):
    '''
    Calculates quantity based on Cq_mean and standard curve
    Params
        plate_df: output from combine_replicates(); df containing Cq_mean
            must be single plate with single target
        std_curve_intercept: output from process_standard()
        std_curve_slope: output from process_standard()
    Returns
        unknown_df: the unknown subset of plate_df, with column Quantity_mean
        Cq_of_lowest_sample_quantity: the Cq value of the lowest pt used on the plate
        intraassay_var intraassay variation (arithmetic mean of the coefficient of variation for all replicates on a plate)
    '''
    if len(plate_df.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')

    unknown_df = plate_df[plate_df.Task == 'Unknown'].copy()

    # define outputs
    unknown_df['Quantity_mean'] = np.nan
    intraassay_var = np.nan
    Cq_of_lowest_sample_quantity = np.nan

    # use standard curve to calculate the quantities for each sample from Cq_mean
    unknown_df['Quantity_mean'] = 10**((unknown_df['Cq_mean'] - std_curve_intercept)/std_curve_slope)

    # calculate the coefficient of variation using QuantStudio original quantities to capture variation on the plate
    if not unknown_df.Q_init_std.isna().all():
        percent_cv = (unknown_df['Q_init_std']-1)*100
        intraassay_var = np.nanmean(percent_cv)

    # Set the Cq of the lowest std quantity for different situations
    if (len(unknown_df.Task) > 0) and not (unknown_df.Cq_mean.isna().all()):
        # plate contains unknowns and at least some have Cq_mean
        Cq_of_lowest_sample_quantity = np.nanmax(unknown_df.Cq_mean)

    return(unknown_df, intraassay_var, Cq_of_lowest_sample_quantity)


def process_ntc(plate_df):
    if len(plate_df.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')

    ntc = plate_df[plate_df.Task == 'Negative Control']
    ntc_is_neg = False
    ntc_Cq = np.nan
    if ntc.is_undetermined.all():
        ntc_is_neg = True
    else:
        if all(ntc.Cq.isna()): # this case should never happen
            ntc_is_neg = np.nan # change NaN to None for dtype consistency
        else:
            ntc_Cq = np.nanmin(ntc.Cq)
    return(ntc_is_neg, ntc_Cq)


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

def process_qpcr_raw(qpcr_raw,include_LoD=False,cutoff=0.9):
    '''wrapper to process whole sheet at once by plate_id and Target
    params
    qpcr_raw: df from read_qpcr_data()
    optional:
    include_LoD adds blod column and moves lowest standard quantity and Cq of lowest standard quantity based on LoD
    cutoff is the fraction of positive replicates in standard curves allowed to consider that standard curve point detectable (used if previous is true)
    '''

    std_curve_df = []
    qpcr_processed = []
    raw_outliers_flagged_df = []
    for [plate_id, target], df in qpcr_raw.groupby(["plate_id", "Target"]):

        ntc_is_neg, ntc_Cq = process_ntc(df)
        outliers_flagged, no_outliers_df = combine_triplicates(df)

        # define outputs and fill with default values
        num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity,lowest_std_quantity,lowest_std_quantity2nd, Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd,slope, intercept, r2, efficiency = np.nan, np.nan,np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # if there are >3 pts in std curve, calculate stats and recalculate quants
        num_points = no_outliers_df[no_outliers_df.Task == 'Standard'].drop_duplicates('Sample').shape[0]
        num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity, lowest_std_quantity, lowest_std_quantity2nd,Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd, slope, intercept, r2, efficiency = process_standard(no_outliers_df)
        std_curve_info = [num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity,lowest_std_quantity, lowest_std_quantity2nd,Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd,slope, intercept, r2, efficiency]
        unknown_df = process_unknown(no_outliers_df, std_curve_info)
        std_curve_df.append([plate_id, target, num_points,  Cq_of_lowest_std_quantity, Cq_of_2ndlowest_std_quantity,lowest_std_quantity,lowest_std_quantity2nd, Cq_of_lowest_std_quantity_gsd, Cq_of_2ndlowest_std_quantity_gsd, slope, intercept, r2, efficiency, ntc_is_neg, ntc_Cq])
        qpcr_processed.append(unknown_df)
        raw_outliers_flagged_df.append(outliers_flagged)

    # compile into dataframes
    raw_outliers_flagged_df = pd.concat(raw_outliers_flagged_df)
    assay_assessment_df=determine_samples_BLoD(raw_outliers_flagged_df, cutoff)
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
                                                        'ntc_is_neg',
                                                        'ntc_Cq'])
    qpcr_processed = pd.concat(qpcr_processed)
    qpcr_processed = qpcr_processed.merge(std_curve_df, how='left', on=['plate_id', 'Target'])
    qpcr_processed,dilution_expts_df = process_dilutions(qpcr_processed)

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

    return(qpcr_processed, std_curve_df, dilution_expts_df,raw_outliers_flagged_df)
