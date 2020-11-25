import pandas as pd
import numpy  as np
from scipy import stats as sci
import warnings

def get_extraction_control(qpcr_averaged, control_sample_code='control_control_PBS'):
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
    df_with_extraction_control = pd.DataFrame()

    for [batch, Target], df in qpcr_averaged.groupby(['batch', 'Target']):
        # set default values
        extraction_control_is_neg = None
        extraction_control_Cq = np.nan

        # if batch has control
        if control_sample_code in df.sample_code.to_list():
            extraction_controls = df[df.sample_code == control_sample_code]

            # if at least one control was not undetermined in all 3 replicates
            if extraction_controls.nondetect_count.min() < 3:
                extraction_control_is_neg = False
                extraction_control_Cq = extraction_controls.Cq_init_mean.min()
            else:
                extraction_control_is_neg = True

        # create columns in df with extraction control info and save
        df['extraction_control_is_neg'] = extraction_control_is_neg
        df['extraction_control_Cq'] = extraction_control_Cq
        df_with_extraction_control =  pd.concat([df_with_extraction_control, df])
    return(df_with_extraction_control)


def process_xeno_inhibition(qpcr_processed_dilutions_xeno, plate_target_info, plate_id, dCt_cutoff=1):
    '''
    takes single plate with Target == Xeno
    finds NTC for that plate in the plate_target_info table, calculates gmean of Cq for Xeno
    determines if samples were inhibited relative to NTC, based on the dCt_cutoff

    Params
    qpcr_processed_dilutions_xeno: processed qPCR data still containing dilutions as separate rows, filtered by Target == 'Xeno'
    plate_target_info: table containing info about plates, including NTCs and their Cqs
    plate_id: the plate_id (coming from the groupby that feeds plates into this function)
    dCt_cutoff: difference between Cq of a sample and Cq of the NTC that defines whether the sample was inhibited, 1 is rule of thumb

    Returns
    qpcr_processed_dilutions_inhibition: dataframe with columns ['Sample', 'dilution', 'xeno_dCt', 'is_inhibited', 'plate_id']
        to be merged with qpcr_processed_dilutions, indicating inhibition
    '''

    # get the geometric mean of the Cqs for the NTCs on the plate

    xeno_ntc_Cq_mean = np.nan
    xeno_ntc = plate_target_info[(plate_target_info.plate_id == plate_id) & (plate_target_info.Target == 'Xeno')]
    if (len(xeno_ntc) > 0) and (~xeno_ntc.ntc_Cq.isna().all()): # skip if there is no NTC with Xeno
        xeno_ntc_Cq_list = xeno_ntc.ntc_Cq.values[0]
        xeno_ntc_Cq_mean = sci.gmean(xeno_ntc_Cq_list) # could use get_gmean() since it ignores nan, but Xeno should never be NaN

    qpcr_processed_dilutions_xeno['xeno_dCt'] = qpcr_processed_dilutions_xeno.Cq_mean - xeno_ntc_Cq_mean
    qpcr_processed_dilutions_xeno['is_inhibited'] = None
    qpcr_processed_dilutions_xeno.loc[qpcr_processed_dilutions_xeno.xeno_dCt >= dCt_cutoff, 'is_inhibited'] = True
    qpcr_processed_dilutions_xeno.loc[qpcr_processed_dilutions_xeno.xeno_dCt < dCt_cutoff, 'is_inhibited'] = False

    qpcr_processed_dilutions_inhibition = qpcr_processed_dilutions_xeno[['Sample', 'dilution', 'xeno_dCt', 'is_inhibited', 'plate_id']]

    return(qpcr_processed_dilutions_inhibition)
