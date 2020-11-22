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


def get_pass_grubbs_test(plate_df, groupby_list, max_std_for_2_reps=0.2, alpha=0.025):
    '''
    max_std_for_2_reps value is from https://www.gene-quantification.de/dhaene-hellemans-qc-data-2010.pdf
    '''

    # make new df
    plate_df_with_grubbs_test = pd.DataFrame()

    # iterate thru the dataframe, grouped by Sample
    for groupby_list, df in plate_df.groupby(groupby_list, as_index=False):
        # make new column 'pass_grubbs_test' that includes the results of the test
        df['pass_grubbs_test'] = None

        if (len(df.Cq.dropna())<3): #cannot evaluate for fewer than 3 values
            if (len(df.Cq.dropna())==2) & (np.std(df.Cq.dropna()) < max_std_for_2_reps):
                df['pass_grubbs_test'] = True
            else:
                df['pass_grubbs_test'] = False
        else:

            b = list(df.Cq) #grubbs takes unindexed list
            outliers = grubbs.max_test_outliers(b, alpha)
            if len(outliers) > 0:
                df['pass_grubbs_test'] = True
                df.loc[df.Cq.isin(outliers), 'pass_grubbs_test'] = False
            else:
                df['pass_grubbs_test'] = True

        plate_df_with_grubbs_test = plate_df_with_grubbs_test.append(df)

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

def combine_triplicates(plate_df_in):
    '''
    Flag outliers via Grubbs test
    Calculate the Cq means, Cq stds, counts before & after removing outliers

    # TODO save triplicates in a new field

    Params
    plate_df_in:
        qpcr data in pandas df, must be 1 plate with 1 target
        should be in the format from QuantStudio3 with
        columns 'Target', 'Sample', 'Cq'

    Returns
    plate_df: same data, with additional column from Grubb's test
        Cq_mean (calculated mean of Cq after excluding outliers)
        Q_QuantStudio_std (calculated standard deviation based on QuantStudio output) for intrassay coefficient of variation
    '''

    if len(plate_df_in.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')

    plate_df = plate_df_in.copy() # fixes pandas warnings

    groupby_list = ['plate_id', 'Sample', 'sample_full','Sample_plate',
                    'Target','Task', 'inhibition_testing','is_dilution',"dilution"]

    # make copy of Cq column and turn this to np.nan for outliers
    # so that they will not be included in the aggregations below
    plate_df['Cq_copy'] = plate_df['Cq'].copy()
    plate_df = get_pass_grubbs_test(plate_df, ['sample_full'])
    plate_df.loc[plate_df.pass_grubbs_test == False, 'Cq_copy'] = np.nan

    # summarize to get mean, std, counts with and without outliers removed

    plate_df_avg = plate_df.groupby(groupby_list).agg(
                                               template_volume=('template_volume','max'),
                                               Q_init_mean=('Quantity','max'), #only needed to preserve quantity information for standards later
                                               Q_init_std=('Quantity', lambda x: np.nan if ( (len(x.dropna()) <2 )| all(np.isnan(x)) ) else (sci.gstd(x.dropna(),axis=0))),
                                               # Q_QuantStudio_std = ('Quantity', 'std'),
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
        sort_b=standard_df.sort_values(by='log_Quantity',ascending=True).copy().reset_index()
        lowest_std_quantity2nd= 10**(sort_b.log_Quantity[1])
        slope, intercept, r2, efficiency = (np.nan, np.nan, np.nan, np.nan)

        if num_points > 2:
            lowest_std_quantity = 10**min(standard_df.log_Quantity)
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
    unknown_df = plate_df[plate_df.Task == 'Unknown'].copy()
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

    # if Cq_mean is zero, don't calculate a quantity (turn to NaN)
    unknown_df.loc[unknown_df[unknown_df.Cq_mean == 0].index, 'Quantity_mean'] = np.nan
    unknown_df['q_diff'] = unknown_df['Q_init_mean'] - unknown_df['Quantity_mean']

    return(unknown_df)

def process_ntc(plate_df):
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


def determine_samples_BLoD(raw_outliers_flagged_df, cutoff):
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
        dfm=dfm[dfm.pass_grubbs_test==True].copy()


        dfm=dfm[dfm.Task=='Standard'] #only standards
        dfm=dfm[dfm.Quantity!=0] #no NTCs
        assay_assessment_df=pd.DataFrame(columns=["Target","LoD_Cq","LoD_Quantity"]) #empty dataframe with desired columns

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
                assay_assessment_df=assay_assessment_df.append(pd.DataFrame({'Target':target, "LoD_Cq": np.nan, "LoD_Quantity":np.nan}), ignore_index=True)

            #usual case for N1/ bCov
            fin=out[out.fr_pos==min(out.fr_pos)].copy()
            if len(fin.fr_pos) ==1:
                assay_assessment_df=assay_assessment_df.append(pd.DataFrame({'Target':target, "LoD_Cq": fin.Cq_mean, "LoD_Quantity":fin.Quantity}), ignore_index=True)
            #usual case for PMMoV/18S
            elif len(fin.fr_pos)>1:
                fin=out[(out.fr_pos==min(out.fr_pos))&(out.Quantity==min(out.Quantity))].copy()
                assay_assessment_df=assay_assessment_df.append(pd.DataFrame({'Target':target, "LoD_Cq": fin.Cq_mean, "LoD_Quantity":fin.Quantity}), ignore_index=True)
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
