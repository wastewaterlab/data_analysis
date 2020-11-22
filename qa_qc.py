import pandas as pd
import numpy  as np
import warnings

def get_extraction_control(qpcr_averaged):
    '''
    determine whether negative controls in each batch were negative (extraction_control_is_neg)
    if negative controls were positive, show the value (extraction_control_Cq)
    if negative controls were missing, return values as None and NaN

    # Xeno must be removed before running this or else all negative controls will appear positive

    Params
    dataframe with columns:
        batch
        sample_code
        is_undetermined_count
        Cq_init_min # should we be using Cq_mean?
    '''
    if 'Xeno' in qpcr_averaged.Target.to_list():
        warnings.warn('Xeno must be removed before this function is run')

    # make empty dataframe to return
    df_with_extraction_control = pd.DataFrame()

    for batch, df in qpcr_averaged.groupby('batch'):
        # set default values
        extraction_control_is_neg = None
        extraction_control_Cq = np.nan

        # if batch has control
        if 'control_control_PBS' in df.sample_code.to_list():
            extraction_controls = df[df.sample_code == 'control_control_PBS']

            # if at least one control was not undetermined in all 3 replicates
            if extraction_controls.is_undetermined_count.min() < 3:
                extraction_control_is_neg = False
                extraction_control_Cq = extraction_controls.Cq_mean.min()
            else:
                extraction_control_is_neg = True

        # create columns in df with extraction control info and save
        df['extraction_control_is_neg'] = extraction_control_is_neg
        df['extraction_control_Cq'] = extraction_control_Cq
        df_with_extraction_control =  pd.concat([df_with_extraction_control, df])
    return(df_with_extraction_control)
