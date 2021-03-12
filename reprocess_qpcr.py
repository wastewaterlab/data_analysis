import numpy as np
import pandas as pd
from scikit_posthocs import outliers_grubbs
from scipy import stats as sci
from sklearn.metrics import r2_score
import warnings


def get_gstd(replicates):
    '''
    compute geometric standard deviation, ignoring NaNs

    Params
    replicates: list of replicate floats (Quantities from qPCR)

    Returns
    gstd_value: geometric standard deviation (float)
    '''
    gstd_value = np.nan
    replicates_no_nan = [x for x in replicates if ~np.isnan(x)]
    if len(replicates_no_nan) >= 2:
        gstd_value = sci.gstd(replicates_no_nan)
    return gstd_value


def get_gmean(replicates):
    '''
    compute geometric mean, ignoring NaNs

    Params
    replicates: list of replicate floats (Quantities from qPCR)

    Returns
    gmean_value: geometric mean (float)
    '''
    gmean_value = np.nan
    replicates_no_nan = [x for x in replicates if ~np.isnan(x)]
    if len(replicates_no_nan) >= 1:
        gmean_value = sci.gmean(replicates_no_nan)
    return gmean_value


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

    return slope, intercept, r2, efficiency


def process_standard(plate_df, target, duplicate_max_std=0.5):
    '''
    from single plate with single target, calculate standard curve info

    Params:
        plate_df: output from combine_replicates(); df containing Cq_mean
            must be single plate with single target
    Returns
        dataframe with columns:
            num_points: number of points used in new std curve
            slope: slope of std curve equation
            intercept: intercept of std curve equation
            r2: r2 of linear regression to make std curve
            efficiency: qPCR efficiency (based on slope, see eq in compute_linear_info)
            loq_Cq: the Cq for the limit of quantification
            loq_Quantity: the quantity for the limit of quantification
            used_default_curve: whether the default curve was used (boolean)
    '''
    # defaults
    std_curve_N1_default = {'slope': -3.446051, 'intercept': 37.827506} # from plate 1093
    std_curve_PMMoV_default = {'slope': -3.548825, 'intercept': 42.188174} # from plate 1092
    std_curve_bCoV_default = {'slope': -3.414, 'intercept': 39.894} # from plate 1353

    # define outputs
    num_points = np.nan
    slope = np.nan
    intercept = np.nan
    r2 = np.nan
    efficiency = np.nan
    loq_Cq = np.nan
    loq_Quantity = np.nan
    used_default_curve = False

    standard_df = plate_df[plate_df.Task == 'Standard'].copy()

    # combine replicates by Quantity
    standard_df = standard_df[['Sample', 'dilution', 'Cq', 'Quantity', 'is_undetermined']]
    standard_df = standard_df.groupby('Quantity').agg(lambda x: x.tolist())
    standard_df = standard_df.reset_index()

    # create summary columns describing all replicates
    standard_df['Cq_no_outliers'] = standard_df.Cq.apply(lambda x: outliers_grubbs(x, alpha=0.05).tolist())

    # np.nanmean etc. will warn if all reps are nan, but still return nan so it's fine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # data required to filter points from curve:
        standard_df['Cq_init_std'] = standard_df.Cq.apply(np.nanstd)
        standard_df['replicate_count'] = standard_df.Cq_no_outliers.apply(lambda x: sum(~np.isnan(x)))
        standard_df['nondetect_count'] = standard_df.is_undetermined.apply(sum)

        # data required for compute_linear_info
        standard_df['Cq_mean'] = standard_df.Cq_no_outliers.apply(np.nanmean)
        standard_df['log_Quantity'] = np.log10(standard_df['Quantity'])

    # rules for including a point on the standard curve:
    # must have amplified for at least 2 of 3 technical replicates
    # if a point amplified for only 2 of 3 replicates, must have std < duplicate_max_std
    df_2reps = standard_df[(standard_df.nondetect_count == 1) & (standard_df.Cq_init_std < duplicate_max_std)]
    df_3reps = standard_df[standard_df.nondetect_count == 0]
    standard_df = pd.concat([df_2reps, df_3reps])
    # catch-all to make sure no points are included that have no Cq_mean or have just 1 replicate after outlier removal
    standard_df = standard_df[(~standard_df.Cq_mean.isna()) & (standard_df.replicate_count > 1)]

    num_points = len(standard_df)

    if num_points > 2:
        slope, intercept, r2, efficiency = compute_linear_info(standard_df)

        # find the lowest quantity and report it and its Cq_mean as the LoQ
        loq_Quantity = standard_df.Quantity.min()
        loq_Cq = standard_df.Cq_mean[standard_df.Quantity.idxmin()]

    def does_slope_have_property(slope):
        '''checks if std curve is missing or poor'''
        return np.isnan(slope) or not -4.0 <= slope <= -3.0 or np.isnan(intercept) or not 30 <= intercept <= 50

    # if std curve is missing or poor, replace with defaults
    if target == 'N1':
        if does_slope_have_property(slope):
            slope = std_curve_N1_default['slope']
            intercept = std_curve_N1_default['intercept']
            used_default_curve = True
    elif target == 'PMMoV':
        if does_slope_have_property(slope):
            slope = std_curve_PMMoV_default['slope']
            intercept = std_curve_PMMoV_default['intercept']
            used_default_curve = True

    elif target == 'bCoV':
        if does_slope_have_property(slope):
            slope = std_curve_bCoV_default['slope']
            intercept = std_curve_bCoV_default['intercept']
            used_default_curve = True

    # save info as a dataframe to return
    std_curve = [num_points, slope, intercept, r2, efficiency,
                 loq_Cq, loq_Quantity,
                 used_default_curve]
    std_curve_cols = ['num_points', 'slope', 'intercept', 'r2', 'efficiency',
                      'loq_Cq', 'loq_Quantity',
                      'used_default_curve']
    std_curve = pd.DataFrame.from_records([std_curve], columns=std_curve_cols)
    return std_curve


def process_unknown(plate_df, intercept, slope, lod=4):
    '''
    Tests for outliers in technical triplicate qPCR reactions
    Converts Cqs to quantities using standard curve intercept and slope
    Substitutes half the LoD for non-detects in technical triplicates
    Calculates geometric mean of substituted quantities for technical triplicates
    Calculates geometric standard dev of non-substituted quantities for technical replicates
    Summarizes non-detects, total replicates reported, and the intraassay variation among triplicates on the plate

    Params
        plate_df: dataframe of raw qPCR data with technical replicates named identically
            must be single plate with single target
        intercept: output from process_standard()
        slope: output from process_standard()
        lod: experimentally determined limit of detection for qPCR assay
    Returns
        unknown_df: the unknown subset of plate_df, with column Quantity_mean
        intraassay_var intraassay variation (arithmetic mean of the coefficient of variation for all replicates on a plate)
    '''

    def Cq_to_quantity(Cq, slope, intercept):
        '''convert Cq to quantity using standard curve'''
        quantity = 10**((Cq - intercept)/slope)
        return quantity

    half_lod = lod / 2

    # filter plate for just unknowns, stop if plate has no unknowns
    unknown_df = plate_df[plate_df.Task == 'Unknown'].copy()
    unknown_df = unknown_df.drop(columns=['Task'])
    if len(unknown_df) == 0:
        return unknown_df, 0

    # define outputs
    intraassay_var = np.nan
    unknown_df['below_limit_of_detection'] = True

    # flatten to columns of lists from long form at this point because outliers_grubbs works on a list
    # and doesn't report the indices for points that get dropped
    unknown_df = unknown_df[['Sample', 'dilution', 'Cq', 'Quantity', 'is_undetermined']]
    unknown_df = unknown_df.groupby(['Sample', 'dilution']).agg(lambda x: x.tolist())
    unknown_df = unknown_df.reset_index()

    # drop outliers from lists, return new column of lists of Cqs without outliers
    unknown_df['Cq_no_outliers'] = unknown_df.Cq.apply(lambda x: outliers_grubbs(x, alpha=0.05).tolist())

    # convert each Cq to quantity, return new column of lists of quantities
    unknown_df['Quantity_no_outliers'] = unknown_df.Cq_no_outliers.apply(lambda x: [Cq_to_quantity(Cq, slope, intercept) for Cq in x])

    #substitute half_lod quantity for NaN (nondetect technical replicates) in lists of quantitites
    unknown_df['Quantity_lod_sub'] = unknown_df.Quantity_no_outliers.apply(lambda x: [half_lod if np.isnan(quant) else quant for quant in x])

    # calculate geometric mean of quantities (after substitution)
    unknown_df['Quantity_mean'] = unknown_df.Quantity_lod_sub.apply(sci.gmean)

    # calculate geometric std of quantities for getting intraassay_var below
    # (without substitution so we can get at true variation, not biased by substitution)
    unknown_df['Quantity_std_nosub'] = unknown_df.Quantity_no_outliers.apply(get_gstd)

    # count replicates used to calculate reported quantities
    unknown_df['replicate_count'] = unknown_df.Cq_no_outliers.apply(lambda x: sum(~np.isnan(x)))

    # count nondetects using the is_undetermined column created by read_gsheets.read_qpcr_data()
    unknown_df['nondetect_count'] = unknown_df.is_undetermined.apply(sum)

    # determine if samples are below the limit of quantification
    unknown_df.loc[unknown_df.Quantity_mean > lod, 'below_limit_of_detection'] = False
    unknown_df.loc[unknown_df.Quantity_mean <= lod, 'below_limit_of_detection'] = True

    # calculate the coefficient of variation using QuantStudio original quantities to capture variation on the plate
    if not unknown_df.Quantity_std_nosub.isna().all():
        percent_cv = (unknown_df['Quantity_std_nosub']-1)*100
        intraassay_var = np.nanmean(percent_cv)

    return unknown_df, intraassay_var


def process_ntc(plate_df, plate_id):
    # ideally there should be only one way of marking a negative control, but data entry is inconsistent
    ntc = plate_df[(plate_df.Task == 'Negative Control') | (plate_df.Sample == 'NTC')]
    ntc_is_neg = False
    ntc_Cq = np.nan

    if len(ntc) == 0:
        warnings.warn(f'Plate {plate_id} is missing NTC')
        return None, np.nan

    if all(ntc.is_undetermined):
        ntc_is_neg = True
    else:
        ntc_Cq = ntc.Cq.mean()
    return ntc_is_neg, ntc_Cq


def process_qpcr_plate(plates, duplicate_max_std=0.5, lod=4):
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
        Target_full = df.Target_full.unique().tolist()
        std_curve = process_standard(df, target, duplicate_max_std)
        ntc_is_neg, ntc_Cq = process_ntc(df, plate_id)
        unknown_df, intraassay_var = process_unknown(df,
                                                   std_curve.intercept.values[0],
                                                   std_curve.slope.values[0],
                                                   lod)

        unknown_df['plate_id'] = plate_id
        unknown_df['Target'] = target
        # save processed unknowns
        qpcr_processed.append(unknown_df)

        # save all info about plate x Target:
        plate_attributes = std_curve.copy() # start from std_curve dataframe
        plate_attributes['plate_id'] = plate_id
        plate_attributes['Target'] = target
        plate_attributes['intraassay_var'] = intraassay_var
        plate_attributes['ntc_is_neg'] = ntc_is_neg
        plate_attributes['ntc_Cq'] = ntc_Cq
        plate_attributes['Target_full'] = [Target_full] # can be more than one, so need to save as list
        plate_target_info.append(plate_attributes)

    # concatenate dataframes for all plate x targets
    qpcr_processed = pd.concat(qpcr_processed)

    plate_target_info = pd.concat(plate_target_info)

    return qpcr_processed, plate_target_info


def choose_dilution(qpcr_processed):
    '''
    RT-qPCR is run at 1x (undiluted) and 5x dilutions for each sample and each assay
    This function multiplies the quantity by the dilution factor and
    finds the least inhibited, most reliably quantified dilution to report for each sample and assay

    NOTE: requires non-duplicated data (each sample run once per assay on a single plate)
    If duplicates exist at the level of ['Sample', 'Target', 'dilution']
    (i.e. the sample was rerun on more than one plate), will warn and then deduplicate.

    Params
    qpcr_processed: dataframe of processed qPCR data with replicates collapsed
    but multiple dilutions possible for each sample x target

    Returns
    qpcr_processed_dilutions: dataframe of processed qPCR data with only the
    best dilution
    '''

    # multiply quantity times dilution to get undiluted_quantity
    qpcr_processed['Quantity_mean_undiluted'] = qpcr_processed['Quantity_mean'] * qpcr_processed['dilution']

    keep_df = []
    for [Sample, Target], df in qpcr_processed.groupby(['Sample', 'Target']):

        # check for duplicates, warn and keep just the first one - data should be clean and this shouldn't happen.
        if len(df.dilution.unique()) != len(df.dilution):
            plate_ids = df.plate_id.unique()
            warnings.warn(f'Sample {Sample} x {Target} has multiple entries with the same dilution factor in plates {plate_ids}')
            df = df.drop_duplicates('dilution', keep='first').copy()

        # if all dilutions were below the limit of detection or were unquantified
        if df.below_limit_of_detection.all() or df.Quantity_mean_undiluted.isna().all():
            # if they both have the same number of nondetects
            if len(df.nondetect_count.unique()) == 1:
                # keep the lowest dilution because this is most likely to be accurately quantified
                keep = df.loc[[df.dilution.idxmin()]]
            # else keep the one with the fewest nondetects
            else:
                keep = df.loc[[df.nondetect_count.idxmin()]]
        # if one of the dilutions was above the limit of detection
        elif len(df[df.below_limit_of_detection == False]) == 1:
            # keep the one that was above limit of quantification
            keep = df[df.below_limit_of_detection == False]
        # if multiple were above limit of detection
        else:
            # choose max Quantity_mean_undiluted
            # if there are multiple dilutions that have the same value for
            # Quantity_mean_undiluted, takes first; nearly impossible for real data
            keep = df.loc[[df.Quantity_mean_undiluted.idxmax()]]

        keep_df.append(keep)

    qpcr_processed_dilutions = pd.concat(keep_df)
    # rename such that we now use the effective Quantity (accounting for
    # dilution) as Quantity_mean going forward
    qpcr_processed_dilutions = qpcr_processed_dilutions.rename(columns = {'Quantity_mean': 'Quantity_mean_with_dilution',
                                               'Quantity_mean_undiluted': 'Quantity_mean'})

    return qpcr_processed_dilutions
