import pandas as pd
import numpy as np
#from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from pandas.core.frame import DataFrame


@dataclass
class ScoringInfo:
    score:float = 0.0
    flag:Optional[str] = None
    point_deduction:Optional[str] = None
    estimation:Optional[str] = None
    def to_tuple(self):
        return (self.score, self.flag, self.point_deduction, self.estimation)


# Sampling parameters

def sample_collectionQC(Sample_type, total_hrs_sampling, sampling_notes) -> ScoringInfo:
    '''Number of hours represented by composite
    Grab sample may be taken if autosampler failed
    # composite_hrsQC('Composite', 24, '')
    # composite_hrsQC('Grab', np.nan, '')
    # composite_hrsQC('Composite', 9, 'sampler failed after 9 hours')
    '''

    si = ScoringInfo()
    name = 'composite_hrs'

    if pd.isnull(Sample_type):
        si.flag = f'missing sample type, but given full points'
        si.score = 1 # missing info, can't score, assume full points
        return si

    if not pd.isnull(sampling_notes):
        si.flag = f'check sampling notes'

    if Sample_type == 'Grab':
        si.score = 0
        si.point_deduction = 'Grab sample instead of composite'
    elif Sample_type == 'Composite':
        # check how many hours are represented (all locations use 24-hr composites)
        if total_hrs_sampling >= 20:
            si.score = 1
        elif total_hrs_sampling >= 10:
            si.score = 0.5
            si.point_deduction = 'Composite sample represents 10-20 hrs, not 24'
        else:
            si.score = 0
            si.point_deduction = 'Composite sample represents < 10 hrs'
    return si


def sample_hold_timeQC(date_extract, date_sampling) -> ScoringInfo:
    '''calculate and account for sample hold time'''

    si = ScoringInfo()
    name = 'sample_hold_time'

    # check if dates are missing from data
    if pd.isnull(date_extract):
        # should actually clean the data so this doesn't need to be here
        si.flag = 'missing date_extract'
        return si

    if pd.isnull(date_sampling):
        si.flag = 'missing check date_sampling'
        return si

    # check sample hold time
    hold_time = date_extract - date_sampling

    if hold_time < np.timedelta64(0, 'D'):
        si.flag = 'date_extract before date_sampling'
    elif hold_time <= np.timedelta64(3, 'D'):
        si.score = 1
    elif hold_time <= np.timedelta64(5, 'D'):
        si.score = 0.5
        si.point_deduction = 'hold time 3-5 days'
    else:
        si.score = 0
        si.point_deduction = 'hold time > 5 days'
        si.estimation = 'under'

    return si


# Extraction parameters

def extraction_neg_controlQC(extraction_control_is_neg, extraction_control_Cq, sample_Cqs) -> ScoringInfo:
    '''Account for results of extraction control'''

    si = ScoringInfo()
    name = 'extraction negative control'

    if extraction_control_is_neg is None:
        si.flag = f'missing {name}'
        si.point_deduction = f'missing {name}'
        si.score = 0 # missing info so can't score
        return si

    if extraction_control_is_neg == True: #good
        si.score = 1
        return si

    else:
        mean_Cq = np.nanmean(sample_Cqs)
        if np.isnan(mean_Cq) or float(extraction_control_Cq) > (mean_Cq + 1):
            # the extraction control amplified but was least 1 Ct higher than
            # the mean sample Cq or sample didn't amplify
            si.score = 0.5
            si.point_deduction = f'{name} amplified, >1 Cq above sample'

        else: #poor
            si.score = 0
            si.point_deduction = f'{name} amplified within 1 Cq of sample'
            si.estimation = 'over'

    return si


def extraction_processing_errorQC(processing_error) -> ScoringInfo:
    '''
    include information about sample processing errors:
    (clogged, cracked spilled, missing, see_notes)
    '''

    si = ScoringInfo()
    name = 'extraction processing error'

    if not pd.isnull(processing_error):
        si.score =  0
        si.flag = 'extraction processing error'
        si.point_deduction = f'extraction processing error: sample {processing_error}'
        si.estimation = 'under'
    else:
        si.score =  1

    return si


def extraction_recovery_controlQC(bCoV_perc_recovered) -> ScoringInfo:
    '''
    Account for recovery efficiency of spike-in control virus
    '''

    si = ScoringInfo()
    name = 'Spike-in control virus recovery'

    if pd.isnull(bCoV_perc_recovered):
        si.flag = f'missing {name}'
        si.score = 0 # no bCoV data, so can't score
        si.point_deduction = f'missing {name}'
        return si

    if bCoV_perc_recovered >= 5:
        si.score = 1
    elif bCoV_perc_recovered >= 1:
        si.score = 0.5
        si.point_deduction = f'{name} was 1-5%'
        si.estimation = 'under'
    else:
        si.score = 0
        si.point_deduction = f'{name} was less than 1%'
        si.estimation = 'under'
    return si


def extraction_fecal_controlQC(pmmov_gc_per_mL) -> ScoringInfo:
    '''
    Check quantity of human fecal indicator virus as a positive control
    '''

    si = ScoringInfo()
    name = 'human fecal indicator virus'

    if np.isnan(pmmov_gc_per_mL):
        si.flag = f'missing {name}'
        si.score = 0 # missing data so can't score
        si.point_deduction = f'missing {name}'
        return si

    if pmmov_gc_per_mL >= 2e3:
        si.score = 1
    elif pmmov_gc_per_mL >= 2e2:
        si.score = 0.5
        si.point_deduction = f'{name} was low'
        si.estimation = 'under'
    else:
        si.score = 0
        si.point_deduction = f'{name} was very low'
        si.estimation = 'under'
    return si


# qPCR parameters
def qpcr_neg_controlQC(ntc_is_neg, ntc_Cq, sample_Cqs) -> ScoringInfo:
    '''given ntc_is_neg, ntc_Cq (no-template control outcomes)
    and list of weights and points
    return quality_score'''


    si = ScoringInfo()
    name = 'no-template qPCR control'

    if pd.isnull(ntc_is_neg):
        si.flag = f'missing {name}'
        si.point_deduction = f'missing {name}'
        return si

    if ntc_is_neg == True: #good
        si.score = 1
        return si
    else:
        mean_Cq = np.nanmean(sample_Cqs)
        if np.isnan(mean_Cq) or float(ntc_Cq) > (mean_Cq + 1):
            # the ntc amplified but was least 1 Ct higher than
            # the mean sample Cq or sample didn't amplify
            si.score = 0.5
            si.point_deduction = f'{name} amplified, >1 Cq above sample'

        else: #poor
            si.score = 0
            si.point_deduction = f'{name} amplified within 1 Cq of sample'
            si.estimation = 'over'

    return si


def qpcr_efficiencyQC(param: float) -> ScoringInfo:
    '''given efficiency (standard curve efficiency) and list of weights and points
    return quality_score'''


    si = ScoringInfo()
    name = 'efficiency'

    if pd.isnull(param):
        si.flag = f'missing {name}'
        si.point_deduction = f'missing {name}'
        return si

    if 0.9 <= param <= 1.1:  # good
        si.score = 1
    elif 0.8 <= param < 0.9:  # ok
        si.score = 0.5
        si.point_deduction = f'{name} 80-90%'
        si.estimation = 'under'
    elif 1.1 < param <= 1.2:  # ok
        si.score = 0.5
        si.point_deduction = f'{name} 110-120%'
        si.estimation = 'over'
    elif param < 0.8: # poor
        si.score = 0
        si.point_deduction = f'{name} < 80%'
        si.estimation = 'under'
    elif param > 1.2: # poor
        si.score = 0
        si.point_deduction = f'{name} > 120%'
        si.estimation = 'over'

    return si


def qpcr_num_pointsQC(param: float) -> ScoringInfo:
    '''given num_points (number of points in the standard curve)
    and list of weights and points
    return quality_score'''


    si = ScoringInfo()
    name = 'points in std curve'

    if pd.isnull(param) or param == 0:
        si.flag = f'missing {name}'
        si.point_deduction = f'missing {name}'
        return si

    if (param >=5): #good
        si.score = 1
    elif (param >=3): #ok
        si.score = 0.5
        si.point_deduction = f'{name} was 3-5'
    else: #poor
        si.score = 0
        si.point_deduction = f'{name} < 3'

    return si


# def qpcr_stdev_techrepsQC(Quantity_std_nosub) -> ScoringInfo:
#     '''Account for geometric standard deviation between quantities per well
#     in technical replicates, as calculated based on standard curve
#     '''
#
#     si = ScoringInfo()
#     name = 'geometric std between technical reps'
#
#     if np.isnan(Quantity_std_nosub):
#         # this will happen for all non-detects
#         si.point_deduction = f'missing {name} due to nondetects'
#         si.score = 1 # give full points
#         return si
#
#     if (Quantity_std_nosub < 2): #good
#         si.score = 1
#     elif (Quantity_std_nosub <= 4): #ok
#         si.score = 0.5
#         si.point_deduction = f'{name} between 2 and 4'
#     else: #poor
#         si.score = 0
#         si.point_deduction = f'{name} greater than 4'
#
#     return si
def qpcr_stdev_techrepsQC(Cq_no_outliers) -> ScoringInfo:
    '''Account for standard deviation between Cqs of
    in technical replicates (after outlier removal)
    '''

    si = ScoringInfo()
    name = 'std of Cqs among technical reps'
    std = np.std(Cq_no_outliers)

    if np.isnan(std):
        # this will happen for all non-detects
        si.point_deduction = f'missing {name} due to nondetects'
        si.score = 1 # give full points
        return si

    if (std < 0.5): #good
        si.score = 1
    elif (std <= 1): #ok
        si.score = 0.5
        si.point_deduction = f'{name} between 0.5 and 1 Cq'
    else: #poor
        si.score = 0
        si.point_deduction = f'{name} greater than 1 Cq'

    return si


def qpcr_inhibitionQC(ratio5x_1x) -> ScoringInfo:
    '''If 5x and 1x dilutions were run and choose_dilution() was used,
    this function will evaluate the 'ratio5x_1x' parameter
    '''

    si = ScoringInfo()
    name = 'qPCR inhibition'

    if pd.isnull(ratio5x_1x): # poor
        # this will happen if one or both dilutions are below limit of detection
        # or if only one dilution was run
        si.score = 0 # can't determine inhibition, may be inhibited!
        si.point_deduction = f'{name} was undetermined'
    elif ratio5x_1x < 2: #good
        si.score = 1.0
    elif ratio5x_1x >= 2: #ok
        si.score = 0.5
        si.point_deduction = f'{name} occurred'

    return si


# parameters not in use

def qpcr_inhibitionQC_original(is_inhibited) -> ScoringInfo:
    '''If multiple dilutions were run and choose_dilution() was used,
    this function will evaluate the 'is_inhibited' parameter
    '''


    si = ScoringInfo()
    name = 'qPCR inhibition'

    if pd.isnull(is_inhibited): # poor
        # this will happen if one or both dilutions are below limit of detection
        # or if only one dilution was run
        si.score = 0 # can't determine inhibition, may be inhibited!
        si.point_deduction = f'{name} was undetermined'
    elif is_inhibited == False: #good
        si.score = 1.0
    elif is_inhibited == True: #ok
        si.score = 0.5
        si.point_deduction = f'{name} was present'

    return si


def qpcr_r2Q(param: float) -> ScoringInfo:
    '''given r2 (qPCR standard curve r2) and list of weights and points
    return quality_score'''

    si = ScoringInfo()
    name = 'r2'

    if np.isnan(param):
        flag = f'check {name}'
        return si

    if (param >=0.98): #good
        si.score = 1
    elif (param >=0.9): #ok
        si.score = 0.5
        si.point_deduction = f'{name} ok'
    else: #poor
        si.score = 0
        si.point_deduction = f'{name} poor'

    return si


def qpcr_num_techrepsQ(param, nondetect_count) -> ScoringInfo:
    '''
    given replicate_count (number of technical replicates passing outlier test),
    nondetect_count (number of true undetermined values in triplicates)
    and list of weights and points
    return quality_score
    '''

    si = ScoringInfo()
    name = 'num tech reps'

    if np.isnan(param):
        flag = f'check {name}'
        return si

    if (param >= 3): #good
        si.score = 1
    elif (param == 2): #ok
        si.score = 0.5
        si.point_deduction = f'{name} = 2'
    elif (param == 1): #poor
        si.score = 0
        si.point_deduction = f'{name} = 1'
    # check if it was a true non-detect; TODO think more about this
    elif (param == 0 ):
        if (nondetect_count >= 3):
            si.score = 1
        elif (nondetect_count == 2):
            si.score = 0.5
            si.point_deduction = 'no reps passed outlier test and number of technical replicates was 2, not 3'
        elif (nondetect_count == 1):
            si.score = 0
            si.point_deduction = 'no reps passed outlier test and number of technical replicates was 1, not 3'

    return si


def get_weights(weights_dict):
    '''
    get weights from dict to df, check dict is complete
    '''
    if weights_dict is None:
        weights_dict = {
        'sample_collection':[5],
        'sample_hold_time':[10],
        'extraction_neg_control':[10],
        'extraction_processing_error':[10],
        'extraction_recovery_control':[10],
        'extraction_fecal_control':[10],
        'qpcr_neg_control':[10],
        'qpcr_efficiency':[10],
        'qpcr_num_points':[10],
        'qpcr_stdev_techreps':[10],
        'qpcr_inhibition':[5],
        }
    else:
        check_keys = {'sample_collection',
        'sample_hold_time',
        'extraction_neg_control',
        'extraction_processing_error',
        'extraction_recovery_control',
        'extraction_fecal_control',
        'qpcr_neg_control',
        'qpcr_efficiency',
        'qpcr_num_points',
        'qpcr_stdev_techreps',
        'qpcr_inhibition'}
        if not all(key in weights_dict.keys() for key in check_keys):
            raise ValueError('missing keys in score_dict')

    # max_score = sum(score_dict.values()) * 3 # max points per param = 3
    # make into dataframe with columns 'parameter', 'weight'
    weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=['weight'])
    if weights_df.weight.sum() != 100:
        raise ValueError('Weights must sum to 100')

    return weights_df

def quality_score(df, weights_dict=None):
    '''
    given a dataframe with all data from wastewater testing
    and a dictionary of scoring values
    calculate a quality score for each data point

    Params
    df : dataframe with columns:
        Sample
        Target
        plate_id
        Sample_type
        total_hrs_sampling
        date_extract
        date_sampling
        extraction_control_is_neg
        extraction_control_Cq
        sample_Cqs
        processing_error
        bCoV_perc_recovered
        pmmov_gc_per_mL
        ntc_is_neg
        ntc_Cq
        efficiency
        num_points
        Cq_no_outliers
        is_inhibited

    weights_dict

    goes thru each data point (row), scores each parameter, applies weights
    sums score for that row and concatenates flags and point_deductions
    normalizes all scores to max_score
    '''

    # check that all necessary fields exist in the DataFrame
    qc_fields = {'Sample',
        'Target',
        'plate_id',
        'Sample_type',
        'total_hrs_sampling',
        'sampling_notes',
        'date_extract',
        'date_sampling',
        'extraction_control_is_neg',
        'extraction_control_Cq',
        'Cq',
        'processing_error',
        'bCoV_perc_recovered',
        'pmmov_gc_per_mL',
        'ntc_is_neg',
        'ntc_Cq',
        'efficiency',
        'num_points',
        'Cq_no_outliers',
        'is_inhibited'}

    if qc_fields - set(df.columns) != set():
        missing = list(qc_fields - set(df.columns))
        raise ValueError(f'dataframe is missing fields: {missing}')

    weights_df = get_weights(weights_dict)
    weights_df = weights_df.reset_index()

    final_scores_df = []
    all_scores_df = []
    for r in df.itertuples():
        # make empty score dataframe for this row

        # call each scoring function and save results in score_df
        sample_collection = sample_collectionQC(r.Sample_type, r.total_hrs_sampling, r.sampling_notes).to_tuple()
        sample_hold_time = sample_hold_timeQC(r.date_extract, r.date_sampling).to_tuple()
        extraction_neg_control = extraction_neg_controlQC(r.extraction_control_is_neg, r.extraction_control_Cq, r.Cq).to_tuple()
        extraction_processing_error = extraction_processing_errorQC(r.processing_error).to_tuple()
        extraction_recovery_control = extraction_recovery_controlQC(r.bCoV_perc_recovered).to_tuple()
        extraction_fecal_control = extraction_fecal_controlQC(r.pmmov_gc_per_mL).to_tuple()
        qpcr_neg_control = qpcr_neg_controlQC(r.ntc_is_neg, r.ntc_Cq, r.Cq).to_tuple()
        qpcr_efficiency = qpcr_efficiencyQC(r.efficiency).to_tuple()
        qpcr_num_points = qpcr_num_pointsQC(r.num_points).to_tuple()
        qpcr_stdev_techreps = qpcr_stdev_techrepsQC(r.Cq_no_outliers).to_tuple()
        qpcr_inhibition = qpcr_inhibitionQC(r.ratio5x_1x).to_tuple()

        # combine all scores for this row into single dataframe
        score_df = [sample_collection,
                    sample_hold_time,
                    extraction_neg_control,
                    extraction_processing_error,
                    extraction_recovery_control,
                    extraction_fecal_control,
                    qpcr_neg_control,
                    qpcr_efficiency,
                    qpcr_num_points,
                    qpcr_stdev_techreps,
                    qpcr_inhibition]
        score_df = pd.DataFrame.from_records(score_df,
                                             columns=['score',
                                                      'flag',
                                                      'point_deduction',
                                                      'estimation'],
                                             index = ['sample_collection',
                                                         'sample_hold_time',
                                                         'extraction_neg_control',
                                                         'extraction_processing_error',
                                                         'extraction_recovery_control',
                                                         'extraction_fecal_control',
                                                         'qpcr_neg_control',
                                                         'qpcr_efficiency',
                                                         'qpcr_num_points',
                                                         'qpcr_stdev_techreps',
                                                         'qpcr_inhibition'])

        # calculate final score, combine all flags and all point deductions
        score = 0
        flags = ''
        point_deductions = ''
        estimation = ''

        # save scoring info in DataFrame
        score_info = score_df[['score']].copy().transpose()
        score_info['Sample'] = r.Sample
        all_scores_df.append(score_info)

        # multiply scores by weights and calculate total weighted scores
        score_df = score_df.reset_index()
        score_df = score_df.merge(weights_df, left_on='index', right_on='index')
        score_df['weighted_score'] = score_df.weight * score_df.score
        quality_score = score_df.weighted_score.sum()

        if len(score_df.flag.dropna()) > 0:
            flags = '; '.join(score_df.flag.dropna())
        if len(score_df.point_deduction.dropna()) > 0:
            point_deductions = '; '.join(score_df.point_deduction.dropna())
        if len(score_df.estimation.dropna()) > 0:
            estimation = '; '.join(score_df.estimation.dropna())
        results = [r.Sample,
                   r.Target,
                   r.plate_id,
                   quality_score,
                   flags,
                   point_deductions,
                   estimation]

        # save final score for the row
        final_scores_df.append(results)

    final_scores_df = pd.DataFrame.from_records(final_scores_df,
                                                columns=['Sample',
                                                         'Target',
                                                         'plate_id',
                                                         'quality_score',
                                                         'flag',
                                                         'point_deduction',
                                                         'estimation'])

    all_scores_df = pd.concat(all_scores_df)
    return(final_scores_df, all_scores_df)
