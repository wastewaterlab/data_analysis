from reprocess_qpcr import *
import numpy as np
import pandas as pd
import pytest


def assert_data_frames_similar(df1, df2, rtol:float=1e-2):
    assert all(df1.columns==df2.columns)
    assert df1.shape==df2.shape, "Shapes differed"
    for colname in df1.columns:
        if df1[colname].dtype in {np.float32, np.float64}:
            assert np.allclose(df1[colname], df2[colname], rtol=rtol), f"Column {colname} differed"
        else:
            assert all(df1[colname]==df2[colname]), f"Column {colname} differed"


def test_get_gmean():
    a = get_gmean([100, 1000, 10000])
    b = (100*1000*10000)**(1/3)
    np.testing.assert_approx_equal(a, b, significant=1)

    a = get_gmean([100, np.nan, 10000])
    b = (100*10000)**(1/2)
    assert round(a, 1) == round(b, 1)

    a = get_gmean([100, np.nan, np.nan])
    assert a == pytest.approx(100.0)

    a = get_gmean([np.nan, np.nan, np.nan])
    assert np.isnan(a) == True


def test_get_gstd():
    a = get_gstd([100, 1000, 10000])
    assert round(a) == 10.0

    a = get_gstd([np.nan, 1000, 10000])
    assert round(a) == 5.0

    a = get_gstd([np.nan, np.nan, 1000])
    assert np.isnan(a) == True

    a = get_gstd([np.nan, np.nan, np.nan])
    assert np.isnan(a) == True


def test_combine_replicates():
    # test1, single sample in triplicate, all amplified
    plate_df = pd.DataFrame({'Sample': ['test1', 'test1', 'test1'],
                         'dilution': [1, 1, 1],
                         'Target': ['N1', 'N1', 'N1'],
                         'Task': ['Unknown', 'Unknown', 'Unknown'],
                         'Cq': [32.25, 32.26, 30],
                         'Quantity': [100, 100, 1000],
                         'is_undetermined': [False, False, False]
                        })
    df = combine_replicates(plate_df, collapse_on=['Sample', 'dilution', 'Task'])



    result_frame1 = pd.DataFrame({
        "Sample": ['test1'],
        "dilution": [1],
        "Task": ['Unknown'],
        "Cq": [[32.25, 32.26, 30]],
        "Quantity": [[100, 100, 1000]],
        "is_undetermined": [[False, False, False]],
        "Cq_no_outliers": [[32.25, 32.26]],
        "Cq_init_mean": [np.mean([32.25, 32.26, 30])],
        "Cq_init_std": [np.std([32.25, 32.26, 30])],
        "Cq_init_min": [30.0],
        "replicate_init_count": [3],
        "Q_init_mean": [sci.gmean([100, 100, 1000])],
        "Q_init_std": [sci.gstd([100, 100, 1000])],
        "Cq_mean": [np.mean([32.25, 32.26])], #[32.255],
        "Cq_std": [np.std([32.25, 32.26])], #[0.005],
        "replicate_count": [2],
        "nondetect_count": [0],
    })

    assert_data_frames_similar(df, result_frame1)

    # test2, single sample in triplicate, none amplified
    plate_df = pd.DataFrame({'Sample': ['test2', 'test2', 'test2'],
                         'dilution': [1, 1, 1],
                         'Target': ['N1', 'N1', 'N1'],
                         'Task': ['Unknown', 'Unknown', 'Unknown'],
                         'Cq': [np.nan, np.nan, np.nan],
                         'Quantity': [np.nan, np.nan, np.nan],
                         'is_undetermined': [True, True, True]
                        })
    df = combine_replicates(plate_df, collapse_on=['Sample', 'dilution', 'Task'])




def test_compute_linear_info():
    plate_data = pd.DataFrame({'Cq_mean': [34.97, 30.79, 27.46, 24.05, 20.66],
                               'log_Quantity': [1.0, 2.0, 3.0, 4.0, 5.0]})
    slope, intercept, r2, efficiency = compute_linear_info(plate_data)
    assert round(slope, 3) == -3.536
    assert round(intercept, 3) == 38.194
    assert round(r2, 3) == 0.998
    assert round(efficiency, 3) == 0.918


def test_process_standard():
    # 1. typical standard curve (>=3 points, ideally 7 points though)
    standard_df = pd.DataFrame({'Task': ['Standard', 'Standard', 'Standard'],
                                'nondetect_count': [0, 0, 0],
                                'Cq_init_std': [0.1, 0.1, 0.1],
                                'Cq_mean': [27.46, 24.05, 20.66],
                                'replicate_count': [3, 3, 3],
                                'Q_init_mean': [1000, 10000, 100000],
                                'replicate_init_count': [3, 3, 3]})
    std_curve_df = process_standard(standard_df,
                                    'N1',
                                    loq_min_reps=(2/3),
                                    duplicate_max_std=0.2)

    assert std_curve_df.num_points.values[0] == 3
    assert std_curve_df.used_default_curve.values[0] == False
    assert round(std_curve_df.slope.values[0], 1) == -3.4
    assert round(std_curve_df.intercept.values[0], 2) == 37.66
    assert round(std_curve_df.r2.values[0], 2) == 1.00
    assert round(std_curve_df.efficiency.values[0], 2) == 0.97
    assert round(std_curve_df.Cq_of_lowest_std_quantity.values[0], 2) == 27.46
    assert round(std_curve_df.loq_Cq.values[0], 2) == 27.46
    assert round(std_curve_df.loq_Quantity.values[0], 2) == 1000

    # 2. standard curve where the lowest point had a non-detect
    # and the Cq_init_std is too big
    # causing the lowest point to be thrown out
    standard_df = pd.DataFrame({'Task': ['Standard', 'Standard', 'Standard', 'Standard'],
                            'nondetect_count': [1, 0, 0, 0],
                            'replicate_count': [3, 3, 3, 3],
                            'Cq_init_std': [0.2, 0.1, 0.1, 0.1],
                            'Cq_mean': [30.79, 27.46, 24.05, 20.66],
                            'Q_init_mean': [100, 1000, 10000, 100000],
                            'replicate_init_count': [3, 3, 3, 3]
                           })
    std_curve_df = process_standard(standard_df,
                                    'N1',
                                    loq_min_reps=(2/3),
                                    duplicate_max_std=0.2)

    assert std_curve_df.num_points.values[0] == 3
    assert std_curve_df.used_default_curve.values[0] == False
    assert round(std_curve_df.slope.values[0], 1) == -3.4
    assert round(std_curve_df.intercept.values[0], 2) == 37.66
    assert round(std_curve_df.r2.values[0], 2) == 1.00
    assert round(std_curve_df.efficiency.values[0], 2) == 0.97
    assert round(std_curve_df.Cq_of_lowest_std_quantity.values[0], 2) == 27.46
    assert round(std_curve_df.loq_Cq.values[0], 2) == 27.46
    assert round(std_curve_df.loq_Quantity.values[0], 2) == 1000

    # 3. curve with just 2 points, which is too few
    # replaced with default standard curve
    standard_df = pd.DataFrame({'Task': ['Standard', 'Standard'],
                                'nondetect_count': [0, 0],
                                'replicate_count': [3, 3],
                                'Cq_init_std': [0.1, 0.1],
                                'Cq_mean': [24.05, 20.66],
                                'Q_init_mean': [10000, 100000],
                                'replicate_init_count': [3, 3]
                               })
    std_curve_df = process_standard(standard_df,
                                'N1',
                                loq_min_reps=(2/3),
                                duplicate_max_std=0.2)
    assert std_curve_df.num_points.values[0] == 2
    assert std_curve_df.used_default_curve.values[0] == True
    assert round(std_curve_df.slope.values[0], 1) == -3.4
    assert round(std_curve_df.intercept.values[0], 2) == 37.83
    assert np.isnan(std_curve_df.r2.values[0]) == True
    assert np.isnan(std_curve_df.efficiency.values[0]) == True
    assert np.isnan(std_curve_df.Cq_of_lowest_std_quantity.values[0]) == True
    assert np.isnan(std_curve_df.loq_Cq.values[0]) == True
    assert np.isnan(std_curve_df.loq_Quantity.values[0]) == True

    # 4. same test as 3 but with PMMoV,
    # which has a different default curve
    std_curve_df = process_standard(standard_df,
                                'PMMoV',
                                loq_min_reps=(2/3),
                                duplicate_max_std=0.2)
    assert std_curve_df.num_points.values[0] == 2
    assert std_curve_df.used_default_curve.values[0] == True
    assert round(std_curve_df.slope.values[0], 1) == -3.5
    assert round(std_curve_df.intercept.values[0], 2) == 42.19

    # 5. standard curve where the lowest point had a non-detect and the Cq_init_std is too big
    # causing the lowest point to be thrown out, resulting in just 2 points, which is too few
    # replaced with default standard curve
    standard_df2 = pd.DataFrame({'Task': ['Standard', 'Standard', 'Standard'],
                            'nondetect_count': [1, 0, 0],
                            'Cq_init_std': [0.2, 0.1, 0.1],
                            'Cq_mean': [27.46, 24.05, 20.66],
                            'replicate_count': [3, 3, 3],
                            'Q_init_mean': [1000, 10000, 100000],
                            'replicate_init_count': [3, 3, 3]
                           })
    std_curve_df = process_standard(standard_df2,
                                    'N1',
                                    loq_min_reps=(2/3),
                                    duplicate_max_std=0.2)
    assert std_curve_df.num_points.values[0] == 2
    assert std_curve_df.used_default_curve.values[0] == True
    assert round(std_curve_df.slope.values[0], 1) == -3.4
    assert round(std_curve_df.intercept.values[0], 2) == 37.83
    assert np.isnan(std_curve_df.r2.values[0]) == True
    assert np.isnan(std_curve_df.efficiency.values[0]) == True
    assert np.isnan(std_curve_df.Cq_of_lowest_std_quantity.values[0]) == True
    assert np.isnan(std_curve_df.loq_Cq.values[0]) == True
    assert np.isnan(std_curve_df.loq_Quantity.values[0]) == True


def test_process_unknown():

    # typical plate, with one sample below limit of quantification
    # Negative Control and Standard rows should be filtered out, returning df with 3 Unknowns
    plate_df = pd.DataFrame({'Sample':['test1', 'test2', 'test3', 'NTC', 'std'],
                             'Task': ['Unknown', 'Unknown', 'Unknown', 'Negative Control', 'Standard'],
                             'Cq_mean': [33, 34, 35, np.nan, 10],
                             'Q_init_std': [5, 3, 1, np.nan, 1],
                            })
    unknown_df, intraassay_var, Cq_of_lowest_sample_quantity = process_unknown(plate_df,
                                                                               std_curve_intercept=37,
                                                                               std_curve_slope=-3.4,
                                                                               std_curve_loq_Cq=34)
    assert unknown_df.Quantity_mean.round(2).to_list() == [15.01,  7.63,  3.87]
    assert unknown_df.Task.to_list() == ['Unknown', 'Unknown', 'Unknown']
    assert unknown_df.below_limit_of_quantification.to_list() == [False, False, True]
    assert intraassay_var == 200.0 # np.mean((5-1)*100,(3-1)*100,(1-1)*100)
    assert Cq_of_lowest_sample_quantity == 35

    # empty plate, should run with no errors but return empty values
    plate_df = pd.DataFrame({'Sample':[],
                             'Task': [],
                             'Cq_mean': [],
                             'Q_init_std': [],
                            })
    unknown_df, intraassay_var, Cq_of_lowest_sample_quantity = process_unknown(plate_df,
                                                                               std_curve_intercept=37,
                                                                               std_curve_slope=-3.4,
                                                                               std_curve_loq_Cq=34)

    assert unknown_df.shape == (0,6)
    assert np.isnan(intraassay_var) == True
    assert np.isnan(Cq_of_lowest_sample_quantity) == True


def test_process_ntc():
    # typical case
    plate_df = pd.DataFrame({'Sample':['test1', 'test2', 'test3', 'NTC', 'std'],
                             'Task': ['Unknown', 'Unknown', 'Unknown', 'Negative Control', 'Standard'],
                             'is_undetermined': [[False, False, False],
                                                 [False, False, False],
                                                 [False, False, False],
                                                 [True, True, True],
                                                 [False, False, False]],
                             'Cq_init_mean': [33, 34, 35, np.nan, 10]
                            })
    ntc_is_neg, ntc_Cq = process_ntc(plate_df, 1000)
    assert ntc_is_neg == True
    assert np.isnan(ntc_Cq) == True

    # NTC amplified in some or all of the triplicates
    plate_df = pd.DataFrame({'Sample':['test1', 'test2', 'test3', 'NTC', 'std'],
                             'Task': ['Unknown', 'Unknown', 'Unknown', 'Negative Control', 'Standard'],
                             'is_undetermined': [[False, False, False],
                                                 [False, False, False],
                                                 [False, False, False],
                                                 [False, False, True],
                                                 [False, False, False]],
                             'Cq_init_mean': [33, 34, 35, 39, 10]
                            })
    ntc_is_neg, ntc_Cq = process_ntc(plate_df, 1000)
    assert ntc_is_neg == False
    assert ntc_Cq == 39

    # NTC missing from plate - this shouldn't happen, but it does
    # check that the proper warning is produced
    plate_df = pd.DataFrame({'Sample':['test1', 'test2', 'test3', 'std'],
                             'Task': ['Unknown', 'Unknown', 'Unknown', 'Standard'],
                             'is_undetermined': [[False, False, False],
                                                 [False, False, False],
                                                 [False, False, False],
                                                 [False, False, False]],
                             'Cq_init_mean': [33, 34, 35, 10]
                            })
    with warnings.catch_warnings(record=True) as w:
        ntc_is_neg, ntc_Cq = process_ntc(plate_df, 1000)
        assert str(w[0].message) == 'Plate 1000 is missing NTC'
        assert ntc_is_neg == None
        assert np.isnan(ntc_Cq) == True


def test_choose_dilution():
    ## test basic example
    df1 = pd.DataFrame({'Sample': ['A', 'A'],
                        'Target': ['N1', 'N1'],
                        'dilution': [1, 5],
                        'Quantity_mean': [19, 4],
                        'below_limit_of_quantification': [False, False],
                        'plate_id': [1, 1]
                      })

    df2 = pd.DataFrame({'Sample': ['A'],
                        'Target': ['N1'],
                        'dilution': [5],
                        'Quantity_mean_with_dilution': [4],
                        'below_limit_of_quantification': [False],
                        'plate_id': [1],
                        'Quantity_mean': [20]
                      })

    df_out = choose_dilution(df1)
    # to assert equal, need to reset index, which creates column called 'index' that must be dropped
    assert_data_frames_similar(df_out.reset_index().drop(columns='index'), df2)

    ## test deduplication
    df1 = pd.DataFrame({'Sample': ['A', 'A'],
                        'Target': ['N1', 'N1'],
                        'dilution': [1, 1],
                        'Quantity_mean': [10, 12],
                        'below_limit_of_quantification': [False, False],
                        'plate_id': [1, 2]
                      })

    df2 = pd.DataFrame({'Sample': ['A'],
                        'Target': ['N1'],
                        'dilution': [1],
                        'Quantity_mean_with_dilution': [10],
                        'below_limit_of_quantification': [False],
                        'plate_id': [1],
                        'Quantity_mean': [10]
                      })

    with warnings.catch_warnings(record=True) as w:
        df_out = choose_dilution(df1)
        assert str(w[0].message) == 'Sample A x N1 has multiple entries with the same dilution factor in plates [1 2]'
        assert_data_frames_similar(df_out, df2)

    ## test one dilution below LoQ
    df1 = pd.DataFrame({'Sample': ['A', 'A'],
                        'Target': ['N1', 'N1'],
                        'dilution': [1, 5],
                        'Quantity_mean': [19, 4],
                        'below_limit_of_quantification': [False, True],
                        'plate_id': [1, 1]
                      })

    df2 = pd.DataFrame({'Sample': ['A'],
                        'Target': ['N1'],
                        'dilution': [1],
                        'Quantity_mean_with_dilution': [19],
                        'below_limit_of_quantification': [False],
                        'plate_id': [1],
                        'Quantity_mean': [19]
                      })

    df_out = choose_dilution(df1)
    # to assert equal, need to reset index, which creates column called 'index' that must be dropped
    assert_data_frames_similar(df_out.reset_index().drop(columns='index'), df2)

    ## test both dilutions below LoQ
    df1 = pd.DataFrame({'Sample': ['A', 'A'],
                        'Target': ['N1', 'N1'],
                        'dilution': [1, 5],
                        'Quantity_mean': [19, 4],
                        'below_limit_of_quantification': [True, True],
                        'plate_id': [1, 1]
                      })

    df2 = pd.DataFrame({'Sample': ['A'],
                        'Target': ['N1'],
                        'dilution': [1],
                        'Quantity_mean_with_dilution': [19],
                        'below_limit_of_quantification': [True],
                        'plate_id': [1],
                        'Quantity_mean': [19]
                      })

    df_out = choose_dilution(df1)
    # to assert equal, need to reset index, which creates column called 'index' that must be dropped
    assert_data_frames_similar(df_out.reset_index().drop(columns='index'), df2)
