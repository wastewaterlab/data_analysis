import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

p = {"efficiency":[0.06,1,0.7,0.2],
     "r2":[0.06,1,0.7,0.2],
     "num_std_points":[0.07,1,0.2,0],
     "used_cong_std":[0.04,1,np.nan, 0],
     "no_template_control":[0.1,1,0.8,0],
     "num_tech_reps":[0.2,1,0.8,0],
     "pcr_inhibition":[0.1,1,np.nan,0],
     "sample_storage":[0.09,1,0.8,0],
     "extraction_neg_control":[0.1,1,0.8,0]}

params_considered = ['efficiency', 'r2', 'num_std_points',
                     'used_cong_std', 'no_template_control',
                     'num_tech_reps', 'pcr_inhibition',
                     'sample_storage', 'extraction_neg_control']

'''
for a processed data frame, will give quality score info

params
p is a dictionary indicating point values. All point values have 3 conditions
where C1 meets acceptable qa/qc for that column, C2 is slightly concerning, and C3 is
very concerning. The weight should be between 0 and 1 and compares the dictionary entries to each other
p should be in the format of 'params_considered': [weight, C1, C2, C3].
params_considered is a list of strings in p to include in the analysis

example: ['eff_std', "Rsq_std",'n_std','n_reps','NTC_std','is_inhibited']

df columns (should correspond to columns listed in params_considered):
Target
efficiency
r2
num_points
ntc_result & Cq_of_lowest_std_quantity
replicate_count & is_undetermined_count
is_inhibited
date_extract & date_sampling & stored_minus_80 & stored_minus_20
PBS_result & Cq_of_lowest_std_quantity

result:
returns the same dataframe with the following columns

quality_score: points from p as a percentage

flags: any samples with quality_score = np.nan have
flag(s) indicating why in the flags column

point_deduction: provides reason(s) for point deduction

'''

# go thru each data point (row) and do all params at once
# get weight and points for good, ok, poor
# 1. define function for each component of the r2_score
# 2. iterate thru each row, calling all the functions
# 3. sum the score and add it as a column
#############

def efficiencyQ(param, points_list):
    '''given std curve efficiency and list of weights and points
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
    '''given std curve r2 and list of weights and points
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
    '''given number of points in the standard curve and list of weights and points
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
    '''given number of technical replicates,
    number of true undetermined values in triplicates
    and list of weights and points
    return quality_score'''

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
    '''given no-template control outcome (pos/neg)
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
    '''given number of points in the standard curve and list of weights and points
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
    '''given number of points in the standard curve and list of weights and points
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
    if (date_extract is np.nan) or (date_extract == 0):
        # should actually clean the data so this doesn't need to be here
        flag = 'check date_extract'
        score = np.nan
        return([score, flag, point_deduction])

    if (date_sampling is np.nan) or (date_sampling == 0):
        flag = 'check date_sampling'
        score = np.nan
        return([score, flag, point_deduction])

    # check sample hold time

    hold_time = datetime.strptime(date_extract, '%Y-%m-%d') - datetime.strptime(date_sampling, '%Y-%m-%d')

    if hold_time < timedelta(days=0):
        score = np.nan
        flag = 'date_extract before date_sampling'
    elif hold_time <= timedelta(days=3):
        score = weight*pts_goodQ
    elif hold_time <= timedelta(days=5):
        score = weight*pts_okQ
        point_deduction = 'hold time 3-5 days'
    else:
        score = weight*pts_poorQ
        point_deduction = 'hold time > 5 days'

    return([score, flag, point_deduction])

def extraction_neg_controlQ(param, Cq_of_lowest_std_quantity, points_list):
    '''given number of points in the standard curve and list of weights and points
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

# def calculate_quality_score(p, params_considered, df):
#
#     points = pd.DataFrame.from_dict(p)
#
#     for row in df.itertuples():
#         score_df = []
#
#         results = efficiencyQ(row.efficiency, points.efficiency.tolist())
#         score_df.append(results)
#
#         results = r2Q(row.r2, points.r2.tolist())
#         score_df.append(results)
#
#         results = num_std_pointsQ(row.num_points, points.num_std_points)
#         score_df.append(results)
