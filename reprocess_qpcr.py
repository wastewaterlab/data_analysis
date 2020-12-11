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

def grubbs_test(replicates, alpha=0.025):
    '''
    from list of triplicates, determine passing replicates

    Params
    replicates: list of Cq values for replicates (usually triplicate)
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
    else:
        replicates_out = replicates_no_nan # return the one value left in the list

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


def combine_replicates(plate_df, collapse_on=['Sample', 'dilution', 'Task']):
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
    if len(plate_df.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')

    # collapse replicates
    plate_df = plate_df[['Sample', 'dilution', 'Task', 'Cq', 'Quantity', 'is_undetermined']]
    plate_df = plate_df.groupby(collapse_on).agg(lambda x: x.tolist())
    plate_df = plate_df.reset_index()

    # create summary columns describing all replicates
    plate_df['Cq_no_outliers'] = plate_df.Cq.apply(lambda x: grubbs_test(x))

    # np.nanmean etc. will warn if all reps are nan, but still return nan so it's fine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plate_df['Cq_init_mean'] = plate_df.Cq.apply(np.nanmean)
        plate_df['Cq_init_std'] = plate_df.Cq.apply(np.nanstd)
        plate_df['Cq_init_min'] = plate_df.Cq.apply(np.nanmin)
        plate_df['replicate_init_count'] = plate_df.Cq.apply(len)

        plate_df['Q_init_mean'] = plate_df.Quantity.apply(get_gmean)
        plate_df['Q_init_std'] = plate_df.Quantity.apply(get_gstd)

        plate_df['Cq_mean'] = plate_df.Cq_no_outliers.apply(np.nanmean)
        plate_df['Cq_std'] = plate_df.Cq_no_outliers.apply(np.nanstd)
        plate_df['replicate_count'] = plate_df.Cq_no_outliers.apply(len)
        plate_df['nondetect_count'] = plate_df.is_undetermined.apply(sum)
    plate_df = plate_df.sort_values(['Sample', 'dilution'])
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


def process_standard(plate_df, target, loq_min_reps=(2/3), duplicate_max_std=0.2):
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
        dict containing keys:
            num_points: number of points used in new std curve
            Cq_of_lowest_std_quantity: the Cq value of the lowest pt used in the new std curve
            #lowest_std_quantity: the Quantity value of the lowest pt used in the new std curve
            #Cq_of_lowest_std_quantity_gsd: geometric standard dviation of the Cq of the lowest standard quantity
            slope: slope of std curve equation
            intercept: intercept of std curve equation
            r2: r2 of linear regression to make std curve
            efficiency: qPCR efficiency (based on slope, see eq in compute_linear_info)
            lod_Cq: the Cq for the limit of quantification
            lod_Quantity: the quantity for the limit of quantification
    '''
    # defaults
    std_curve_N1_default = {'num_points': 0,
                            'slope': -3.446051, 'intercept': 37.827506,
                            'r2': np.nan,
                            'efficiency': np.nan,
                            'Cq_of_lowest_std_quantity': np.nan,
                            'loq_Cq': np.nan,
                            'loq_Quantity': np.nan} # from plate 1093
    std_curve_PMMoV_default = {'num_points': 0,
                               'slope': -3.548825, 'intercept': 42.188174,
                               'r2': np.nan,
                               'efficiency': np.nan,
                               'Cq_of_lowest_std_quantity': np.nan,
                               'loq_Cq': np.nan,
                               'loq_Quantity': np.nan} # from plate 1092

    # define outputs
    std_curve = {'num_points': np.nan,
                 'slope': np.nan,
                 'intercept': np.nan,
                 'r2': np.nan,
                 'efficiency': np.nan,
                 'Cq_of_lowest_std_quantity': np.nan,
                 'loq_Cq': np.nan,
                 'loq_Quantity': np.nan}
                 # Cq_of_2ndlowest_std_quantity = np.nan
                 # Cq_of_lowest_std_quantity_gsd = np.nan
                 # Cq_of_2ndlowest_std_quantity_gsd = np.nan
                 # lowest_std_quantity = np.nan
                 # lowest_std_quantity2nd = np.nan
    used_default_curve = False

    standard_df = plate_df[plate_df.Task == 'Standard'].copy()

    # rules for including a point on the standard curve:
    # must have amplified for at least 2 of 3 technical replicates
    # if a point amplified for only 2 of 3 replicates, check that they have std < 0.2 or else remove
    df_2reps = standard_df[(standard_df.nondetect_count == 1) & (standard_df.Cq_init_std < duplicate_max_std)]
    df_3reps = standard_df[standard_df.nondetect_count == 0]
    standard_df = pd.concat([df_2reps, df_3reps])
    # catch-all to make sure no points are included that have no Cq_mean or have just 1 replicate after outlier removal
    standard_df = standard_df[(~standard_df.Cq_mean.isna()) & (standard_df.replicate_count > 1)]

    std_curve['num_points'] = len(standard_df)

    if std_curve['num_points'] > 2:
        standard_df['log_Quantity'] = np.log10(standard_df['Q_init_mean'])
        std_curve['slope'], std_curve['intercept'], std_curve['r2'], std_curve['efficiency'] = compute_linear_info(standard_df)

        standard_df = standard_df.sort_values('Q_init_mean')
        std_curve['Cq_of_lowest_std_quantity'] = standard_df.Cq_mean.values[0]
        # # find the Cq_mean of the lowest and second lowest (for LoQ) standard quantity
        # Cq_of_2ndlowest_std_quantity = standard_df.Cq_mean.values[1]
        # #find the geometric standard deviation of the Cq of the lowest and second lowest (for LoQ) standard quantity
        # Cq_of_lowest_std_quantity_gsd = standard_df.Cq_std.values[0]
        # Cq_of_2ndlowest_std_quantity_gsd = standard_df.Cq_std.values[1]
        # # the lowest and second lowest (for LoQ) standard quantity
        # lowest_std_quantity = standard_df.Q_init_mean.values[0]
        # lowest_std_quantity2nd = standard_df.Q_init_mean.values[1]


    # if curve is poor or missing, replace with defaults
    if target == 'N1':
        if (np.isnan(std_curve['slope'])) or \
           (std_curve['slope'] > -2.5) or \
           (std_curve['slope'] < -5.0) or \
           (np.isnan(std_curve['intercept'])) or \
           (std_curve['intercept'] > 50) or \
           (std_curve['intercept'] < 30):
           std_curve = std_curve_N1_default
           used_default_curve = True
    elif target == 'PMMoV':
        if (np.isnan(std_curve['slope'])) or \
           (std_curve['slope'] > -2.5) or \
           (std_curve['slope'] < -5.0) or \
           (np.isnan(std_curve['intercept'])) or \
           (std_curve['intercept'] > 50) or \
           (std_curve['intercept'] < 30):
           std_curve = std_curve_PMMoV_default
           used_default_curve = True

    ## determine LoQ
    # TODO revisit the definitions of Limits of Quantification vs Detection
    # determine the fraction of the replicates that were detectable
    standard_df['fraction_positive'] = np.nan
    standard_df['fraction_positive'] = standard_df.replicate_count / standard_df.replicate_init_count

    #only take the portion of the dataframe that is >= the cutoff
    standard_df = standard_df[standard_df.fraction_positive >= loq_min_reps].copy()

    # if there are standards that meet the cutoff
    # find the lowest quantity and report it and its Cq_mean as the LoD values
    if len(standard_df) > 0:
        standard_df = standard_df.sort_values('Q_init_mean')
        std_curve['loq_Cq'] = standard_df.Cq_mean.values[0]
        std_curve['loq_Quantity'] = standard_df.Q_init_mean.values[0]

    return(std_curve, used_default_curve)


def process_unknown(plate_df, std_curve_intercept, std_curve_slope, std_curve_loq_Cq):
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

    unknown_df = plate_df[plate_df.Task == 'Unknown'].copy()

    # define outputs
    unknown_df['Quantity_mean'] = np.nan
    intraassay_var = np.nan
    Cq_of_lowest_sample_quantity = np.nan
    unknown_df['below_limit_of_quantification'] = None

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

    # determine if samples are below the limit of quantification
    if std_curve_loq_Cq is not np.nan:
        unknown_df.loc[unknown_df.Cq_mean > std_curve_loq_Cq, 'below_limit_of_quantification'] = True
        unknown_df.loc[unknown_df.Cq_mean <= std_curve_loq_Cq, 'below_limit_of_quantification'] = False

    return(unknown_df, intraassay_var, Cq_of_lowest_sample_quantity)


def process_ntc(plate_df, plate_id):
    # ideally there should be only one way of marking a negative control, but data entry is inconsistent
    ntc = plate_df[(plate_df.Task == 'Negative Control') | (plate_df.Sample == 'NTC')]
    ntc_is_neg = False
    ntc_Cq = np.nan

    if len(ntc) == 0:
        warnings.warn(f'Plate {plate_id} is missing NTC')
        return(None, np.nan)

    if all(ntc.is_undetermined.values[0]): # is_undetermined is a list, need to access the list itself to ask if all values are True
        ntc_is_neg = True
    else:
        if np.isnan(ntc.Cq_init_mean.values[0]): # this case should never happen- if Cq_init_mean is NaN, all values are undetermined
            ntc_is_neg = None
        else:
            ntc_Cq = ntc.Cq_init_mean.values[0]
    return(ntc_is_neg, ntc_Cq)


def process_qpcr_plate(plates, loq_min_reps=(2/3)):
    '''wrapper to process data from a qPCR plate(s) grouped by unique plate_id and Target combo
    Params
        plates: df from read_qpcr_data() containing raw qPCR data.
            NOTE: must have dilution column
        loq_min_reps: fraction of positive replicates in standard curves allowed to consider that standard curve point quantifiable
    Returns
        qpcr_processed: dataframe containing processed qPCR data, replicates collapsed
        plate_target_info:
    '''
    if 'dilution' not in plates.columns:
        raise ValueError(''' qPCR data is missing column 'dilution' ''')

    plate_target_info = []
    qpcr_processed = []

    for [plate_id, target], df in plates.groupby(['plate_id', 'Target']):
    # process plate
        plate_attributes = []

        plate_df = combine_replicates(df)
        std_curve, used_default_curve = process_standard(plate_df, target, loq_min_reps)
        ntc_is_neg, ntc_Cq = process_ntc(plate_df, plate_id)
        unknown_df, intraassay_var, Cq_of_lowest_sample_quantity = process_unknown(plate_df,
                                                                                   std_curve['intercept'],
                                                                                   std_curve['slope'],
                                                                                   std_curve['loq_Cq'])

        unknown_df['plate_id'] = plate_id
        unknown_df['Target'] = target
        # save processed unknowns
        qpcr_processed.append(unknown_df)

        # save all info about plate x Target:
        # TODO find cleaner way to save and make a dataframe that knows about column names
        plate_attributes = [plate_id,
                            target,
                            std_curve['num_points'],
                            std_curve['slope'],
                            std_curve['intercept'],
                            std_curve['r2'],
                            std_curve['efficiency'],
                            std_curve['loq_Cq'],
                            std_curve['loq_Quantity'],
                            std_curve['Cq_of_lowest_std_quantity'],
                            used_default_curve,
                            intraassay_var,
                            Cq_of_lowest_sample_quantity,
                            ntc_is_neg,
                            ntc_Cq]
        #plate_attributes = pd.DataFrame(plate_attributes)
        plate_target_info.append(plate_attributes)

    # concatenate dataframes for all plate x targets
    qpcr_processed = pd.concat(qpcr_processed)
    column_names = ['plate_id',
    'Target',
    'num_points',
    'slope',
    'intercept',
    'r2',
    'efficiency',
    'loq_Cq',
    'loq_Quantity',
    'Cq_of_lowest_std_quantity',
    'used_default_curve',
    'intraassay_var',
    'Cq_of_lowest_sample_quantity',
    'ntc_is_neg',
    'ntc_Cq']

    plate_target_info = pd.DataFrame.from_records(plate_target_info, columns=column_names)

    return(qpcr_processed, plate_target_info)


def choose_dilution(qpcr_processed):
    '''
    multiply quantity by the dilution factor and choose the least inhibited dilution
    also try to choose a dilution for which the raw Cq_mean was above the limit of quantification
    NOTE: requires non-duplicated data. If duplicates exist at the level of ['Sample', 'Target', 'dilution'], will warn and then deduplicate.

    Params
    qpcr_processed: dataframe of processed qPCR data with replicates collapsed but multiple dilutions possible for each sample x target

    Returns
    qpcr_processed_dilutions: dataframe of processed qPCR data with only the best dilution
    '''

    # multiply quantity times dilution to get undiluted_quantity
    qpcr_processed['Quantity_mean_undiluted'] = qpcr_processed['Quantity_mean'] * qpcr_processed['dilution']

    keep = []
    for [Sample, Target], df in qpcr_processed.groupby(['Sample', 'Target']):

        # check for duplicates, warn and keep just the first one - data should be clean and this shouldn't happen.
        if len(df.dilution.unique()) != len(df.dilution):
            plate_ids = df.plate_id.to_list()
            warnings.warn(f'Sample {Sample} x {Target} has multiple entries with the same dilution factor in plates {plate_ids}')
            df = df[df.duplicated('dilution', keep='first')]

        # if none of the dilutions were above limit of quantification
        # keep the 1x dilution
        if len(df[df.below_limit_of_quantification == False]) == 0:
            keep.append(df[df.dilution == 1])
        elif len(df[df.below_limit_of_quantification == False]) == 1:
            # check which one was above limit of quantification and keep that one
            keep.append(df[df.below_limit_of_quantification == False])
        else:
            # if multiple were above limit of detection, then choose max Quantity_mean_undiluted
            keep.append(df[df.Quantity_mean_undiluted == df.Quantity_mean_undiluted.max()])

    qpcr_processed_dilutions = pd.concat(keep)
    qpcr_processed_dilutions = qpcr_processed_dilutions.rename(columns = {'Quantity_mean': 'Quantity_mean_with_dilution',
                                               'Quantity_mean_undiluted': 'Quantity_mean'})

    return(qpcr_processed_dilutions)
