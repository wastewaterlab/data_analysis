import pandas as pd
import numpy  as np
from scipy import stats as sci
import warnings

def get_extraction_control(sample_data_qpcr, control_sample_code='control_control_PBS'):
    '''
    determine whether negative controls in each batch were negative (extraction_control_is_neg)
    if negative controls were positive, show the value (extraction_control_Cq)
    if negative controls were missing, return values as None and NaN

    Params
    dataframe with columns:
        batch
        sample_code
        nondetect_count
        Cq_mean
    '''

    # make empty dataframe to return
    df_with_extraction_control = []
    # save samples missing batch number
    missing_batch = sample_data_qpcr[sample_data_qpcr.batch.isna()]
    df_with_extraction_control.append(missing_batch)

    for [batch, Target], df in sample_data_qpcr.groupby(['batch', 'Target']):
        # set default values
        extraction_control_is_neg = None
        extraction_control_Cq = np.nan

        # if batch has control
        if control_sample_code in df.sample_code.to_list():
            extraction_controls = df[df.sample_code == control_sample_code]

            # if at least one control amplified in 1 of 3 replicates
            # (if none amplified, nondetect_count = 3)
            if extraction_controls.nondetect_count.min() < 3:
                extraction_control_is_neg = False
                # Cq is a list, hence apply to get mean of list
                # then choose lowest extraction control mean Cq from the batch
                extraction_control_Cq = extraction_controls.Cq.apply(np.nanmean).min()
            else:
                extraction_control_is_neg = True

        # create columns in df with extraction control info and save
        df_save = df.copy() # make copy to prevent editing the original
        df_save['extraction_control_is_neg'] = extraction_control_is_neg
        df_save['extraction_control_Cq'] = extraction_control_Cq
        # remove extraction control rows
        df_save = df_save[df_save.sample_code != control_sample_code].copy()
        df_with_extraction_control.append(df_save)
    df_with_extraction_control = pd.concat(df_with_extraction_control)

    return(df_with_extraction_control)
