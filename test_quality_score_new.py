from quality_score_new import *

def test_efficiency():
    assert efficiencyQ(.98, [1,1,0.5,0.25]) == [1, '', '']
    assert efficiencyQ(.75, [1,1,0.5,0.25]) == [0.5, '', 'efficiency ok']
    assert efficiencyQ(.65, [1,1,0.5,0.25]) == [0.25, '', 'efficiency poor']
    assert efficiencyQ(.30, [1,1,0.5,0.25]) == [0, 'set to 0', 'efficiency very poor']
    assert efficiencyQ(np.nan, [1,1,0.5,0.25]) == [0, 'check efficiency', '']

def test_r2Q():
    assert r2Q(.98, [1,1,0.5,0.25]) == [1, '', '']
    assert r2Q(.95, [1,1,0.5,0.25]) == [.5, '', 'r2 ok']
    assert r2Q(.85, [1,1,0.5,0.25]) == [.25, '', 'r2 poor']
    assert r2Q(np.nan, [1,1,0.5,0.25]) == [0, 'check r2', '']

def test_num_std_pointsQ():
    assert num_std_pointsQ(6, [1,1,0.5,0.25]) == [1, '', '']
    assert num_std_pointsQ(4, [1,1,0.5,0.25]) == [0.5, '', 'points in std curve ok']
    assert num_std_pointsQ(2, [1,1,0.5,0.25]) == [0.25, '', 'points in std curve poor']
    assert num_std_pointsQ(0, [1,1,0.5,0.25]) == [0, 'check points in std curve', '']
    assert num_std_pointsQ(np.nan, [1,1,0.5,0.25]) == [0, 'check points in std curve', '']

def test_num_tech_repsQ():
    # (num_reps, is_undetermined_count, points)
    assert num_tech_repsQ(3, 0, [1,1,0.5,0.25]) == [1, '', '']
    assert num_tech_repsQ(2, 0, [1,1,0.5,0.25]) == [0.5, '', 'num tech reps = 2']
    assert num_tech_repsQ(1, 0, [1,1,0.5,0.25]) == [0.25, '', 'num tech reps = 1']
    assert num_tech_repsQ(0, 3, [1,1,0.5,0.25]) == [1, '', '']
    assert num_tech_repsQ(0, 2, [1,1,0.5,0.25]) == [0.5, '', 'no reps passed outlier test and number of technical replicates was 2, not 3']
    assert num_tech_repsQ(0, 1, [1,1,0.5,0.25]) == [0.25, '', 'no reps passed outlier test and number of technical replicates was 1, not 3']
    assert num_tech_repsQ(np.nan, 3, [1,1,0.5,0.25]) == [0, 'check num tech reps', '']

def test_no_template_controlQ():
    # no_template_controlQ(param, Cq_of_lowest_std_quantity, points_list)
    assert no_template_controlQ('negative', 37, [1,1,0.5,0.25]) == [1, '', '']
    assert no_template_controlQ(39, 37, [1,1,0.5,0.25]) == [0.5, '', 'no-template qPCR control had low-level amplification']
    assert no_template_controlQ(38, 37, [1,1,0.5,0.25]) == [0.25, '', 'no-template qPCR control amplified']
    assert no_template_controlQ(np.nan, 37, [1,1,0.5,0.25]) == [0, 'check no-template qPCR control', '']
    assert no_template_controlQ(37, np.nan, [1,1,0.5,0.25]) == [0, 'check Cq_of_lowest_std_quantity', '']

def test_pcr_inhibitionQ():
    assert pcr_inhibitionQ(False, [1,1,0.5,0.25]) == [1, '', '']
    assert pcr_inhibitionQ('No', [1,1,0.5,0.25]) == [1, '', '']
    assert pcr_inhibitionQ(True, [1,1,0.5,0.25]) == [0.25, '', 'sample has PCR inhibition']
    assert pcr_inhibitionQ('Yes', [1,1,0.5,0.25]) == [0.25, '', 'sample has PCR inhibition']
    assert pcr_inhibitionQ('unknown', [1,1,0.5,0.25]) == [0, 'test for inhibition has not been performed', '']
    assert pcr_inhibitionQ(np.nan, [1,1,0.5,0.25]) == [0, 'check PCR inhibition', '']

def test_sample_storageQ():
    # sample_storageQ(date_extract, date_sampling, stored_minus_80, stored_minus_20, points_list)
    assert sample_storageQ('2020-08-01', '2020-07-31', 0, 0, [1,1,0.5,0.25]) == [1, '', ''] # 1 day
    assert sample_storageQ('2020-08-01', '2020-07-28', 0, 0, [1,1,0.5,0.25]) == [0.5, '', 'hold time 3-5 days'] # > 3 days
    assert sample_storageQ('2020-08-01', '2020-07-25', 0, 0, [1,1,0.5,0.25]) == [0.25, '', 'hold time > 5 days'] # > 5 days
    assert sample_storageQ('2020-08-01', '2020-07-31', 0, 1, [1,1,0.5,0.25]) == [0, 'Sample was frozen before extraction', ''] # frozen
    assert sample_storageQ('2020-08-01', '2020-07-31', 1, 0, [1,1,0.5,0.25]) == [0, 'Sample was frozen before extraction', ''] # frozen
    assert sample_storageQ(np.nan, '2020-07-31', 0, 0, [1,1,0.5,0.25]) == [np.nan, 'check date_extract', ''] # missing date
    assert sample_storageQ('2020-08-01', np.nan, 0, 0, [1,1,0.5,0.25]) == [np.nan, 'check date_sampling', ''] # missing date

def test_extraction_neg_controlQ():
    #extraction_neg_controlQ(param, Cq_of_lowest_std_quantity, points_list)
    assert extraction_neg_controlQ('negative', 37, [1,1,0.5,0.25]) == [1, '', '']
    assert extraction_neg_controlQ(39, 37, [1,1,0.5,0.25]) == [0.5, '', 'extraction negative control had low-level amplification']
    assert extraction_neg_controlQ(38, 37, [1,1,0.5,0.25]) == [0.25, '', 'extraction negative control amplified']
    assert extraction_neg_controlQ(np.nan, 37, [1,1,0.5,0.25]) == [0, 'check extraction negative control', '']
    assert extraction_neg_controlQ(37, np.nan, [1,1,0.5,0.25]) == [0, 'check Cq_of_lowest_std_quantity', '']

# run all tests
test_efficiency()
test_r2Q()
test_num_std_pointsQ()
test_num_tech_repsQ()
test_no_template_controlQ()
test_pcr_inhibitionQ()
test_sample_storageQ()
test_extraction_neg_controlQ()
