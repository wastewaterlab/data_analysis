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
    # composite_hrsQC('Composite', 24, '', 1)
    # composite_hrsQC('Grab', np.nan, '', 1)
    # composite_hrsQC('Composite', 9, 'sampler failed after 9 hours', 1)
    '''

    si = ScoringInfo()
    name = 'composite_hrs'

    if Sample_type is np.nan:
        si.flag = f'missing {name}'
        return si

    if sampling_notes != '':
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


def sample_hold_timeQC(date_extract, date_sampling) -> ScoringInfo::
    '''calculate and account for sample hold time'''

    si = ScoringInfo()
    name = 'sample_hold_time'

    # check if dates are missing from data
    if (date_extract is np.nan) or (date_extract == 0) or (pd.isnull(date_extract)):
        # should actually clean the data so this doesn't need to be here
        flag = 'check date_extract'
        return si

    if (date_sampling is np.nan) or (date_sampling == 0) or (pd.isnull(date_sampling)):
        flag = 'check date_sampling'
        return si

    # check sample hold time
    hold_time = date_extract - date_sampling

    if hold_time < np.timedelta64(0, 'D'):
        flag = 'date_extract before date_sampling'
    elif hold_time <= np.timedelta64(3, 'D'):
        si.score = 1
    elif hold_time <= np.timedelta64(5, 'D'):
        si.score = 0.5
        si.point_deduction = 'hold time 3-5 days'
    else:
        si.score = 0
        si.point_deduction = 'hold time > 5 days'

    return si


# Extraction parameters

def extraction_neg_controlQC(extraction_control_is_neg, extraction_control_Cq, sample_Cqs) -> ScoringInfo:
    '''given extraction control info and loq_Cq and list of weights
    return quality_score'''

    si = ScoringInfo()
    name = 'extraction negative control'

    if extraction_control_is_neg is None:
        flag = f'check {name}'
        return si

    if extraction_control_is_neg == True: #good
        si.score = 1
        return si

    else:
        mean_Cq = np.nanmean(sample_Cqs)
        if np.isnan(mean_Cq) or float(ntc_Cq) > (mean_Cq + 1):
            # the extraction control amplified but was least 1 Ct higher than
            # the mean sample Cq or sample didn't amplify
            si.score = 0.5
            si.point_deduction = f'{name} amplified, >1 Cq above sample'

        else: #poor
            si.score = 0
            si.point_deduction = f'{name} amplified within 1 Cq of sample'

    return si


def extraction_processing_errorQC(processing_error) -> ScoringInfo:
    '''
    include information about sample processing errors:
    (clogged, cracked spilled, missing, see_notes)
    extraction_processing_errorQC('clogged', 1)
    extraction_processing_errorQC('', 1)
    extraction_processing_errorQC(np.nan, 1)
    '''

    si = ScoringInfo()
    name = 'extraction processing error'

    if str(processing_error) != 'nan' and str(processing_error) != '':
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
    # extraction_recovery_controlQC(11.5, [1,3,2,1])
    # extraction_recovery_controlQC(5.2, [1,3,2,1])
    # extraction_recovery_controlQC(2.1, [1,3,2,1])
    '''

    si = ScoringInfo()
    name = 'Spike-in control virus recovery'

    # TODO do we want this flag or not?
    if bCoV_perc_recovered is np.nan:
        si.flag = f'missing {name}'
        return si

    if bCoV_perc_recovered >= 10:
        si.score = 1
    elif bCoV_perc_recovered >= 5:
        si.score = 0.5
        si.point_deduction = f'{name} was 5-10%'
        si.estimation = 'under'
    else:
        si.score = 0
        si.point_deduction = f'{name} was less than 5%'
        si.estimation = 'under'
    return si


def extraction_fecal_controlQC(pmmov_gc_per_mL) -> ScoringInfo:
    '''
    Check quantity of human fecal indicator virus as a positive control
    # extraction_fecal_controlQC(99, [1,3,2,1])
    # extraction_fecal_controlQC(105, [1,3,2,1])
    # extraction_fecal_controlQC(10000, [1,3,2,1])
    '''

    si = ScoringInfo()
    name = 'human fecal indicator virus'

    # TODO do we want this flag or not?
    if pmmov_gc_per_mL is np.nan:
        si.flag = f'missing {name}'
        return si

    if pmmov_gc_per_mL >= 1e3:
        si.score = 1
    elif pmmov_gc_per_mL >= 1e2:
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

    if ntc_is_neg is np.nan:
        flag = f'check {name}'
        return si

    if ntc_is_neg: #good
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

    return si


def qpcr_efficiencyQC(param: float) -> ScoringInfo:
    '''given efficiency (standard curve efficiency) and list of weights and points
    return quality_score'''


    si = ScoringInfo()
    name = 'efficiency'

    if param is np.nan:
        flag = f'check {name}'
        return si

    if 0.9 <= param <= 1.1:  # good
        si.score = 1
    elif 0.8 <= param <= 1.2:  # ok
        si.score = 0.5
        si.point_deduction = f'{name} ok'
        si.underestimated = True
    elif 0.7 <= param <= 1.3:  # poor
        si.score = 0
        si.point_deduction = f'{name} poor'
        si.underestimated = True

    return si


def qpcr_num_pointsQC(param: float) -> ScoringInfo:
    '''given num_points (number of points in the standard curve)
    and list of weights and points
    return quality_score'''


    si = ScoringInfo()
    name = 'points in std curve'

    if (param is np.nan) or (param == 0):
        flag = f'check {name}'
        return si

    if (param >=5): #good
        si.score = 1
    elif (param >=3): #ok
        si.score = 0.5
        si.point_deduction = f'{name} ok'
    elif (param < 3): #poor
        si.score = 0
        si.point_deduction = f'{name} poor'

    return si


def qpcr_stdev_techrepsQC(Quantity_std_nosub) -> ScoringInfo:
    '''Account for geometric standard deviation between quantities
    in technical replicates, as calculated based on standard curve
    std_between_techrepsQC(4.5, [1,3,2,1]) # poor
    std_between_techrepsQC(2, [1,3,2,1]) # ok
    std_between_techrepsQC(1, [1,3,2,1]) # good
    '''


    si = ScoringInfo()
    name = 'geometric std between technical reps'

    if Quantity_std_nosub is np.nan:
        # this will happen for all non-detects
        return si

    if (Quantity_std_nosub < 2): #good
        si.score = 1
    elif (Quantity_std_nosub <= 4): #ok
        si.score = 0.5
        si.point_deduction = f'{name} between 2 and 4'
    else: #poor
        si.score = 0
        si.point_deduction = f'{name} greater than 4'

    return si


def qpcr_inhibitionQC(is_inhibited) -> ScoringInfo:
    '''Account for geometric standard deviation between quantities
    in technical replicates, as calculated based on standard curve
    std_between_techrepsQC(4.5, [1,3,2,1]) # poor
    std_between_techrepsQC(2, [1,3,2,1]) # ok
    std_between_techrepsQC(1, [1,3,2,1]) # good
    '''


    si = ScoringInfo()
    name = 'qPCR inhibition'

    if is_inhibited == None: # ok
        # this will happen if one or both dilutions are below limit of detection
        si.score = 0.5
        si.point_deduction = f'{name} was undetermined'
    elif is_inhibited == False: #good
        si.score = 1
    elif is_inhibited == True: #poor
        si.score = 0
        si.point_deduction = f'{name} was present'

    return si


# parameters not in use
def qpcr_r2Q(param: float) -> ScoringInfo:
    '''given r2 (qPCR standard curve r2) and list of weights and points
    return quality_score'''


    si = ScoringInfo()
    name = 'r2'

    if param is np.nan:
        flag = f'check {name}'
        return si

    if (param >=0.98): #good
        si.score = 1
    elif (param >=0.9): #ok
        si.score = 0.5
        si.point_deduction = f'{name} ok'
    elif (param < 0.9): #poor
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

    if param is np.nan:
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
    get weights
    '''
    if weights_dict is None:
        weights_dict = {
        'sample_collection':[5],
        'sample_hold_time':[5],
        'extraction_neg_control':[10],
        'extraction_processing_error':[10],
        'extraction_recovery_control':[10],
        'extraction_fecal_control':[10],
        'qpcr_neg_control':[10],
        'qpcr_efficiency':[10],
        'qpcr_num_points':[10],
        'qpcr_stdev_techreps':[10],
        'qpcr_inhibition':[10],
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
        Quantity_std_nosub
        is_inhibited

    weights_dict

    goes thru each data point (row), scores each parameter, applies weights
    sums score for that row and concatenates flags and point_deductions
    normalizes all scores to max_score
    '''

    weights_df = get_weights(weights_dict)

    final_scores_df = []
    for row in df.itertuples():
        # make empty score dataframe for this row

        # call each scoring function and save results in score_df
        sample_collection = sample_collectionQC(Sample_type, total_hrs_sampling, sampling_notes).to_tuple()
        sample_hold_time = sample_hold_timeQC(date_extract, date_sampling).to_tuple()
        extraction_neg_control = extraction_neg_controlQC(extraction_control_is_neg, extraction_control_Cq, sample_Cqs).to_tuple()
        extraction_processing_error = extraction_processing_errorQC(processing_error).to_tuple()
        extraction_recovery_control = extraction_recovery_controlQC(bCoV_perc_recovered).to_tuple()
        extraction_fecal_control = extraction_fecal_controlQC(pmmov_gc_per_mL).to_tuple()
        qpcr_neg_control = qpcr_neg_controlQC(ntc_is_neg, ntc_Cq, sample_Cqs).to_tuple()
        qpcr_efficiency = qpcr_efficiencyQC(efficiency).to_tuple()
        qpcr_num_points = qpcr_num_pointsQC(num_points).to_tuple()
        qpcr_stdev_techreps = qpcr_stdev_techrepsQC(Quantity_std_nosub).to_tuple()
        qpcr_inhibition = qpcr_inhibitionQC(is_inhibited).to_tuple()

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
                                                      'underestimated'],
                                             index = 'sample_collection',
                                                         'sample_hold_time',
                                                         'extraction_neg_control',
                                                         'extraction_processing_error',
                                                         'extraction_recovery_control',
                                                         'extraction_fecal_control',
                                                         'qpcr_neg_control',
                                                         'qpcr_efficiency',
                                                         'qpcr_num_points',
                                                         'qpcr_stdev_techreps',
                                                         'qpcr_inhibition')

        # calculate final score, combine all flags and all point deductions
        score = 0
        flags = ''
        point_deductions = ''

        weights_df = weights_df.reset_index()
        scores_df = scores_df.reset_index()
        scores_df = scores_df.merge(weights_df, left_on='index', right_on='index')
        scores_df['weighted_score'] = scores_df.weight * scores_df.score
        total_score = scores_df.weighted_score.sum()

        if len(score_df.flag.dropna()) > 0:
            flags = '; '.join(score_df.flag.dropna())
        if len(score_df.point_deduction.dropna()) > 0:
            point_deductions = '; '.join(score_df.point_deduction.dropna())
        results = [row.Sample, row.Target, row.plate_id, score, flags, point_deductions]

        # save final score for the row
        final_scores_df.append(results)

    final_scores_df = pd.DataFrame.from_records(final_scores_df, columns=['Sample', 'Target', 'plate_id', 'score', 'flag', 'point_deduction'])
    final_scores_df['quality_score'] = (final_scores_df.score / max_score)*100

    return(final_scores_df)
