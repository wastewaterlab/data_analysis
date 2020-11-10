import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

def efficiencyQ(param, points_list):
    '''given efficiency (standard curve efficiency) and list of weights and points
    return quality_score'''

    weight = points_list[0]
    pts_goodQ = points_list[1]
    pts_okQ = points_list[2]
    pts_poorQ = points_list[3]

    param_name = 'efficiency'
    score = 0
    flag = np.nan
    point_deduction = np.nan

    if param is np.nan:
        flag = f'check {param_name}'
        return([score, flag, point_deduction])

    if (param >=0.8) & (param <=1.1): #good
        score = weight*pts_goodQ
    elif (param >=0.7) & (param <=1.2): #ok
        score = weight*pts_okQ
        point_deduction = f'{param_name} ok'
    elif (param >=0.6) & (param <=1.3): #poor
        score = weight*pts_poorQ
        point_deduction = f'{param_name} poor'
    elif (param <0.6) | (param >1.3): #very poor
        flag = 'set to 0'
        point_deduction = f'{param_name} very poor'

    return([score, flag, point_deduction])

def r2Q(param, points_list):
    '''given r2 (qPCR standard curve r2) and list of weights and points
    return quality_score'''

    weight = points_list[0]
    pts_goodQ = points_list[1]
    pts_okQ = points_list[2]
    pts_poorQ = points_list[3]

    param_name = 'r2'
    score = 0
    flag = np.nan
    point_deduction = np.nan

    if param is np.nan:
        flag = f'check {param_name}'
        return([score, flag, point_deduction])

    if (param >=0.98): #good
        score = weight*pts_goodQ
    elif (param >=0.9): #ok
        score = weight*pts_okQ
        point_deduction = f'{param_name} ok'
    elif (param < 0.9): #poor
        score = weight*pts_poorQ
        point_deduction = f'{param_name} poor'

    return([score, flag, point_deduction])

def num_std_pointsQ(param, points_list):
    '''given num_points (number of points in the standard curve)
    and list of weights and points
    return quality_score'''

    weight = points_list[0]
    pts_goodQ = points_list[1]
    pts_okQ = points_list[2]
    pts_poorQ = points_list[3]

    param_name = 'points in std curve'
    score = 0
    flag = np.nan
    point_deduction = np.nan

    if (param is np.nan) or (param == 0):
        flag = f'check {param_name}'
        return([score, flag, point_deduction])

    if (param >=5): #good
        score = weight*pts_goodQ
    elif (param >=3): #ok
        score = weight*pts_okQ
        point_deduction = f'{param_name} ok'
    elif (param < 3): #poor
        score = weight*pts_poorQ
        point_deduction = f'{param_name} poor'

    return([score, flag, point_deduction])

def num_tech_repsQ(param, is_undetermined_count, points_list):
    '''
    given replicate_count (number of technical replicates passing outlier test),
    is_undetermined_count (number of true undetermined values in triplicates)
    and list of weights and points
    return quality_score
    '''

    weight = points_list[0]
    pts_goodQ = points_list[1]
    pts_okQ = points_list[2]
    pts_poorQ = points_list[3]

    param_name = 'num tech reps'
    score = 0
    flag = np.nan
    point_deduction = np.nan

    if param is np.nan:
        flag = f'check {param_name}'
        return([score, flag, point_deduction])

    if (param >= 3): #good
        score = weight*pts_goodQ
    elif (param == 2): #ok
        score = weight*pts_okQ
        point_deduction = f'{param_name} = 2'
    elif (param == 1): #poor
        score = weight*pts_poorQ
        point_deduction = f'{param_name} = 1'
    # check if it was a true non-detect; TODO think more about this
    elif (param == 0 ):
        if (is_undetermined_count >= 3):
            score = weight*pts_goodQ
        elif (is_undetermined_count == 2):
            score = weight*pts_okQ
            point_deduction = 'no reps passed outlier test and number of technical replicates was 2, not 3'
        elif (is_undetermined_count == 1):
            score = weight*pts_poorQ
            point_deduction = 'no reps passed outlier test and number of technical replicates was 1, not 3'
        elif (is_undetermined_count == 0):
            flag = 'set to 0'
            point_deduction = '0 replicates'

    return([score, flag, point_deduction])

def no_template_controlQ(param, Cq_of_lowest_std_quantity, points_list):
    '''given ntc_result (no-template control outcome)
    and list of weights and points
    return quality_score'''

    weight = points_list[0]
    pts_goodQ = points_list[1]
    pts_okQ = points_list[2]
    pts_poorQ = points_list[3]

    param_name = 'no-template qPCR control'
    score = 0
    flag = np.nan
    point_deduction = np.nan

    if param is np.nan:
        flag = f'check {param_name}'
        return([score, flag, point_deduction])

    if param =='negative': #good
        score = weight*pts_goodQ
        return([score, flag, point_deduction])

    if Cq_of_lowest_std_quantity is np.nan:
        flag = f'check Cq_of_lowest_std_quantity'
        return([score, flag, point_deduction])

    elif float(param) > (Cq_of_lowest_std_quantity + 1):
        # the ntc amplified but was least 1 Ct higher than the lowest conconcentration point on the standard curve
        score = weight*pts_okQ
        point_deduction = f'{param_name} had low-level amplification'

    else: #poor
        score = weight*pts_poorQ
        point_deduction = f'{param_name} amplified'

    return([score, flag, point_deduction])

def pcr_inhibitionQ(param, points_list):
    '''given is_inhibited and list of weights and points
    return quality_score'''

    weight = points_list[0]
    pts_goodQ = points_list[1]
    pts_okQ = points_list[2]
    pts_poorQ = points_list[3]

    param_name = 'PCR inhibition'
    score = 0
    flag = np.nan
    point_deduction = np.nan

    if (param is np.nan): #should score be na or zero in this case?
        flag = f'check {param_name}'
        return([score, flag, point_deduction])

    if param == 'unknown': #should score be na or zero in this case?
        flag = 'test for inhibition has not been performed'
        return([score, flag, point_deduction])

    if (param is False) or (param == 'No'): #good
        score = weight*pts_goodQ
    else: #poor
        score = weight*pts_poorQ
        point_deduction = f'sample has {param_name}'

    return([score, flag, point_deduction])

def sample_storageQ(date_extract, date_sampling, stored_minus_80, stored_minus_20, points_list):
    '''given date_extract, date_sampling, stored_minus_80, stored_minus_20 and
    list of weights and points
    return quality_score'''

    weight = points_list[0]
    pts_goodQ = points_list[1]
    pts_okQ = points_list[2]
    pts_poorQ = points_list[3]

    param_name = 'sample_storage'
    score = 0
    flag = np.nan
    point_deduction = np.nan

    # check if sample was frozen before extraction
    #(TODO: there should just be one column for all sample storage ['fresh', '4C', '-20', '-80'])
    if (stored_minus_80 == 1) or (stored_minus_20 == 1):
        score = 0
        flag = 'Sample was frozen before extraction'
        return([score, flag, point_deduction])

    # check if dates are missing from data
    if (date_extract is np.nan) or (date_extract == 0) or (date_extract is np.nat):
        # should actually clean the data so this doesn't need to be here
        flag = 'check date_extract'
        score = np.nan
        return([score, flag, point_deduction])

    if (date_sampling is np.nan) or (date_sampling == 0) or (date_sampling is np.nat):
        flag = 'check date_sampling'
        score = np.nan
        return([score, flag, point_deduction])

    # check sample hold time
    hold_time = date_extract - date_sampling

    if hold_time < np.timedelta64(0, 'D'):
        score = np.nan
        flag = 'date_extract before date_sampling'
    elif hold_time <= np.timedelta64(3, 'D'):
        score = weight*pts_goodQ
    elif hold_time <= np.timedelta64(5, 'D'):
        score = weight*pts_okQ
        point_deduction = 'hold time 3-5 days'
    else:
        score = weight*pts_poorQ
        point_deduction = 'hold time > 5 days'

    return([score, flag, point_deduction])

def extraction_neg_controlQ(param, Cq_of_lowest_std_quantity, points_list):
    '''given PBS_result and Cq_of_lowest_std_quantity and list of weights and points
    return quality_score'''

    weight = points_list[0]
    pts_goodQ = points_list[1]
    pts_okQ = points_list[2]
    pts_poorQ = points_list[3]

    param_name = 'extraction negative control'
    score = 0
    flag = np.nan
    point_deduction = np.nan

    if param is np.nan:
        flag = f'check {param_name}'
        return([score, flag, point_deduction])

    if param =='negative': #good
        score = weight*pts_goodQ
        return([score, flag, point_deduction])

    if Cq_of_lowest_std_quantity is np.nan:
        flag = f'check Cq_of_lowest_std_quantity'
        return([score, flag, point_deduction])

    elif float(param) > (Cq_of_lowest_std_quantity + 1):
        # the ntc amplified but was least 1 Ct higher than the lowest conconcentration point on the standard curve
        score = weight*pts_okQ
        point_deduction = f'{param_name} had low-level amplification'

    else: #poor
        score = weight*pts_poorQ
        point_deduction = f'{param_name} amplified'

    return([score, flag, point_deduction])

def get_scoring_matrix(score_dict=None):
    '''
    define the scoring matrix
    input must be either None or a dictionary
    dictionary keys must include: 'efficiency', 'r2', 'num_std_points', 'used_cong_std', 'no_template_control', 'num_tech_reps', 'pcr_inhibition', 'sample_storage', 'extraction_neg_control'
    dictionary values must be a list: [weight, pts_goodQ, pts_okQ, pts_poorQ]
    '''
    p = score_dict
    if score_dict is None:
        p = {"efficiency":[0.06,1,0.7,0.2],
             "r2":[0.06,1,0.7,0.2],
             "num_std_points":[0.07,1,0.2,0],
             "used_cong_std":[0.04,1,np.nan, 0],
             "no_template_control":[0.1,1,0.8,0],
             "num_tech_reps":[0.2,1,0.8,0],
             "pcr_inhibition":[0.1,1,np.nan,0],
             "sample_storage":[0.09,1,0.8,0],
             "extraction_neg_control":[0.1,1,0.8,0]}

    else:
        check_keys = ['efficiency', 'r2', 'num_std_points', 'used_cong_std', 'no_template_control', 'num_tech_reps', 'pcr_inhibition', 'sample_storage', 'extraction_neg_control']
        if not all(key in p.keys() for key in check_keys):
            raise ValueError('missing keys in score_dict')
    points = pd.DataFrame.from_dict(p)
    points['points'] = ['weight', 'pts_goodQ', 'pts_okQ', 'pts_poorQ']


    ## calculate max score by manipulating the points dataframe

    points_t = points.transpose()
    points_t.loc['points'].tolist()
    points_t.columns = points_t.loc['points']
    points_t = points_t.drop('points', axis = 0)
    max_score = (points_t.weight * points_t.pts_goodQ).sum()

    return(points, max_score)

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
        is_undetermined_count
        ntc_result
        Cq_of_lowest_std_quantity
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
        efficiency = efficiencyQ(row.efficiency, points.efficiency.tolist())

        r2 = r2Q(row.r2, points.r2.tolist())

        num_std_points = num_std_pointsQ(row.num_points, points.num_std_points)

        num_tech_reps = num_tech_repsQ(row.replicate_count, row.is_undetermined_count, points.num_tech_reps)

        no_template_control = no_template_controlQ(row.ntc_result, pd.to_numeric(row.Cq_of_lowest_std_quantity), points.no_template_control)

        pcr_inhibition = pcr_inhibitionQ(row.is_inhibited, points.pcr_inhibition)

        sample_storage = sample_storageQ(row.date_extract, row.date_sampling, row.stored_minus_80, row.stored_minus_20, points.sample_storage)

        extraction_neg_control = extraction_neg_controlQ(row.PBS_result, row.Cq_of_lowest_std_quantity, points.extraction_neg_control)

        # combine all scores for this row into single dataframe
        score_df = [efficiency, r2, num_std_points, num_tech_reps,no_template_control,
                    pcr_inhibition, extraction_neg_control] #sample_storage
        score_df = pd.DataFrame.from_records(score_df, columns=['score', 'flag', 'point_deduction'])

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
