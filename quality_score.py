import pandas as pd
import numpy as np
from collections import namedtuple
from datetime import datetime
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from pandas.core.frame import DataFrame


RETURNED_SCORING_INFO = namedtuple("scoring_info", ["score", "flag", "point_deduction", "underestimated"])

def make_default_scoring_info():
    return RETURNED_SCORING_INFO(score=0.0, flag=None, point_deduction=None, underestimated=None)


def efficiencyQ(param: float, points_list) -> RETURNED_SCORING_INFO:
    '''given efficiency (standard curve efficiency) and list of weights and points
    return quality_score'''

    weight, pts_goodQ, pts_okQ, pts_poorQ = points_list
    score, flag, point_deduction, underestimated = make_default_scoring_info()
    name = 'efficiency'

    if param is np.nan:
        flag = f'check {name}'
        return score, flag, point_deduction, underestimated

    if 0.8 <= param <= 1.1:  # good
        score = weight*pts_goodQ
    elif 0.7 <= param <= 1.2:  # ok
        score = weight*pts_okQ
        point_deduction = f'{name} ok'
        underestimated = True
    elif 0.6 <= param <= 1.3:  # poor
        score = weight*pts_poorQ
        point_deduction = f'{name} poor'
        underestimated = True
    elif (param <0.6) | (param >1.3): #very poor
        flag = 'set to 0'
        point_deduction = f'{name} very poor'
        underestimated = True

    return score, flag, point_deduction, underestimated


def r2Q(param: float, points_list) -> RETURNED_SCORING_INFO:
    '''given r2 (qPCR standard curve r2) and list of weights and points
    return quality_score'''

    weight, pts_goodQ, pts_okQ, pts_poorQ = points_list
    score, flag, point_deduction, underestimated = make_default_scoring_info()
    name = 'r2'

    if param is np.nan:
        flag = f'check {name}'
        return score, flag, point_deduction, underestimated

    if (param >=0.98): #good
        score = weight*pts_goodQ
    elif (param >=0.9): #ok
        score = weight*pts_okQ
        point_deduction = f'{name} ok'
    elif (param < 0.9): #poor
        score = weight*pts_poorQ
        point_deduction = f'{name} poor'

    return score, flag, point_deduction, underestimated

def num_std_pointsQ(param, points_list) -> RETURNED_SCORING_INFO:
    '''given num_points (number of points in the standard curve)
    and list of weights and points
    return quality_score'''

    weight, pts_goodQ, pts_okQ, pts_poorQ = points_list
    score, flag, point_deduction, underestimated = make_default_scoring_info()
    name = 'points in std curve'

    if (param is np.nan) or (param == 0):
        flag = f'check {name}'
        return score, flag, point_deduction, underestimated

    if (param >=5): #good
        score = weight*pts_goodQ
    elif (param >=3): #ok
        score = weight*pts_okQ
        point_deduction = f'{name} ok'
    elif (param < 3): #poor
        flag = 'set to 0'
        point_deduction = f'{name} poor'

    return score, flag, point_deduction, underestimated

def num_tech_repsQ(param, nondetect_count, points_list) -> RETURNED_SCORING_INFO:
    '''
    given replicate_count (number of technical replicates passing outlier test),
    nondetect_count (number of true undetermined values in triplicates)
    and list of weights and points
    return quality_score
    '''

    weight, pts_goodQ, pts_okQ, pts_poorQ = points_list
    score, flag, point_deduction, underestimated = make_default_scoring_info()
    name = 'num tech reps'

    if param is np.nan:
        flag = f'check {name}'
        return score, flag, point_deduction, underestimated

    if (param >= 3): #good
        score = weight*pts_goodQ
    elif (param == 2): #ok
        score = weight*pts_okQ
        point_deduction = f'{name} = 2'
    elif (param == 1): #poor
        score = weight*pts_poorQ
        point_deduction = f'{name} = 1'
    # check if it was a true non-detect; TODO think more about this
    elif (param == 0 ):
        if (nondetect_count >= 3):
            score = weight*pts_goodQ
        elif (nondetect_count == 2):
            score = weight*pts_okQ
            point_deduction = 'no reps passed outlier test and number of technical replicates was 2, not 3'
        elif (nondetect_count == 1):
            score = weight*pts_poorQ
            point_deduction = 'no reps passed outlier test and number of technical replicates was 1, not 3'
        elif (nondetect_count == 0):
            flag = 'set to 0'
            point_deduction = '0 replicates'

    return score, flag, point_deduction, underestimated

def no_template_controlQ(ntc_is_neg, ntc_Cq, loq_Cq, points_list) -> RETURNED_SCORING_INFO:
    '''given ntc_is_neg, ntc_Cq (no-template control outcomes)
    and list of weights and points
    return quality_score'''

    weight, pts_goodQ, pts_okQ, pts_poorQ = points_list
    score, flag, point_deduction, underestimated = make_default_scoring_info()
    name = 'no-template qPCR control'

    if ntc_is_neg is np.nan:
        flag = f'check {name}'
        return score, flag, point_deduction, underestimated

    if ntc_is_neg: #good
        score = weight*pts_goodQ
        return score, flag, point_deduction, underestimated

    if loq_Cq is np.nan:
        flag = f'check loq_Cq'
        return score, flag, point_deduction, underestimated

    elif float(ntc_Cq) > (loq_Cq + 1):
        # the ntc amplified but was least 1 Ct higher than the lowest conconcentration point on the standard curve
        score = weight*pts_okQ
        point_deduction = f'{name} had low-level amplification'

    else: #poor
        score = weight*pts_poorQ
        point_deduction = f'{name} amplified'

    return score, flag, point_deduction, underestimated


def sample_storageQ(date_extract, date_sampling, points_list):
    '''given date_extract, date_sampling and
    list of weights and points
    return quality_score'''

    weight, pts_goodQ, pts_okQ, pts_poorQ = points_list
    score, flag, point_deduction, underestimated = make_default_scoring_info()
    name = 'sample_storage'

    # check if sample was frozen before extraction
    #(TODO: there should just be one column for all sample storage ['fresh', '4C', '-20', '-80'])

    # check if dates are missing from data
    if (date_extract is np.nan) or (date_extract == 0) or (pd.isnull(date_extract)):
        # should actually clean the data so this doesn't need to be here
        flag = 'check date_extract'
        return score, flag, point_deduction, underestimated

    if (date_sampling is np.nan) or (date_sampling == 0) or (pd.isnull(date_sampling)):
        flag = 'check date_sampling'
        return score, flag, point_deduction, underestimated

    # check sample hold time
    hold_time = date_extract - date_sampling

    if hold_time < np.timedelta64(0, 'D'):
        flag = 'date_extract before date_sampling'
    elif hold_time <= np.timedelta64(3, 'D'):
        score = weight*pts_goodQ
    elif hold_time <= np.timedelta64(5, 'D'):
        score = weight*pts_okQ
        point_deduction = 'hold time 3-5 days'
    else:
        score = weight*pts_poorQ
        point_deduction = 'hold time > 5 days'

    return score, flag, point_deduction, underestimated

def extraction_neg_controlQ(extraction_control_is_neg, extraction_control_Cq, loq_Cq, points_list):
    '''given extraction control info and loq_Cq and list of weights and points
    return quality_score'''

    weight, pts_goodQ, pts_okQ, pts_poorQ = points_list
    score, flag, point_deduction, underestimated = make_default_scoring_info()
    name = 'extraction negative control'

    if extraction_control_is_neg is None:
        flag = f'check {name}'
        return score, flag, point_deduction, underestimated

    if extraction_control_is_neg: #good
        score = weight*pts_goodQ
        return score, flag, point_deduction, underestimated

    if loq_Cq is np.nan:
        flag = f'check loq_Cq'
        return score, flag, point_deduction, underestimated

    elif float(extraction_control_Cq) > (loq_Cq + 1):
        # the ntc amplified but was least 1 Ct higher than the lowest conconcentration point on the standard curve
        score = weight*pts_okQ
        point_deduction = f'{name} had low-level amplification'

    else: #poor
        score = weight*pts_poorQ
        point_deduction = f'{name} amplified'

    return score, flag, point_deduction, underestimated

def get_scoring_matrix(score_dict:Optional[Dict[str,List[float]]]=None) -> Tuple[DataFrame, float]:
    '''
    define the scoring matrix
    input must be either None or a dictionary
    dictionary keys must include: 'efficiency', 'r2', 'num_std_points', 'used_cong_std', 'no_template_control', 'num_tech_reps', 'sample_storage', 'extraction_neg_control'
    dictionary values must be a list: [weight, pts_goodQ, pts_okQ, pts_poorQ]
    '''
    if score_dict is None:
        score_dict = {"efficiency":[0.06,1,0.7,0.2],
             "r2":[0.06,1,0.7,0.2],
             "num_std_points":[0.07,1,0.2,0],
             #"used_cong_std":[0.04,1,np.nan, 0],
             "no_template_control":[0.1,1,0.8,0],
             "num_tech_reps":[0.2,1,0.8,0],
             "sample_storage":[0.09,1,0.8,0],
             "extraction_neg_control":[0.1,1,0.8,0]}
    else:
        check_keys = {'efficiency', 'r2', 'num_std_points', 'used_cong_std', 'no_template_control', 'num_tech_reps', 'sample_storage', 'extraction_neg_control'}
        if not all(key in score_dict.keys() for key in check_keys):
            raise ValueError('missing keys in score_dict')

    points = pd.DataFrame.from_dict(score_dict)
    points['points'] = ['weight', 'pts_goodQ', 'pts_okQ', 'pts_poorQ']


    ## calculate max score by manipulating the points dataframe

    points_t = points.transpose()
    points_t.loc['points'].tolist()
    points_t.columns = points_t.loc['points']
    points_t = points_t.drop('points', axis = 0)
    max_score = (points_t.weight * points_t.pts_goodQ).sum()

    return points, max_score

def quality_score(df, scoring_dict=None):
    '''
    given a dataframe with all data from wastewater testing
    and a dictionary of scoring values
    calculate a quality score for each data point

    Params
    df : dataframe with columns:
        Sample
        Target
        plate_id
        efficiency
        r2
        num_points
        replicate_count
        nondetect_count
        ntc_is_neg
        ntc_Cq
        loq_Cq
        is_inhibited
        date_extract
        date_sampling
        stored_minus_80
        stored_minus_20
        PBS_result
    score_dict : input for get_scoring_matrix() (see that function)

    goes thru each data point (row) and calculates each component of the score
    sums score for that row and concatenates flags and point_deductions
    normalizes all scores to max_score
    '''

    points, max_score = get_scoring_matrix(scoring_dict)

    final_scores_df = []
    for row in df.itertuples():
        # make empty score dataframe for this row

        # call each scoring function and save results in score_df
        efficiency = tuple(efficiencyQ(row.efficiency, points.efficiency.tolist()))
        r2 = tuple(r2Q(row.r2, points.r2.tolist()))
        num_std_points = tuple(num_std_pointsQ(row.num_points, points.num_std_points))
        num_tech_reps = tuple(num_tech_repsQ(row.replicate_count, row.nondetect_count, points.num_tech_reps))
        no_template_control = tuple(no_template_controlQ(row.ntc_is_neg, row.ntc_Cq, pd.to_numeric(row.loq_Cq), points.no_template_control))
        sample_storage = tuple(sample_storageQ(row.date_extract, row.date_sampling, points.sample_storage))
        extraction_neg_control = tuple(extraction_neg_controlQ(row.extraction_control_is_neg, row.extraction_control_Cq, row.loq_Cq, points.extraction_neg_control))

        # combine all scores for this row into single dataframe
        score_df = [efficiency, r2,
                    num_std_points,
                    num_tech_reps,
                    no_template_control,
                    sample_storage,
                    extraction_neg_control]
        score_df = pd.DataFrame.from_records(score_df, columns=['score', 'flag', 'point_deduction', 'underestimated'])

        # calculate final score, combine all flags and all point deductions
        score = 0
        flags = np.nan
        point_deductions = np.nan
        if 'set to 0' not in score_df.flag:
            score = score_df.score.sum()
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
