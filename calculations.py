import pandas as pd
import numpy as np
from scipy import stats as sci
from scipy.stats.mstats import gmean
from scipy.stats import gstd

def calculate_gc_per_l(qpcr_data, replace_bloq= False ):
    '''
    calculates and returns gene copies / L

    Params
    replace_bloq should the function replace any sample that is below the limit of quantification with the loq
    qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
            gc_per_ul_input
            Quantity_mean
            template_volume
            elution_vol_ul
            weight_vol_extracted_ml
        if replace_bloq is true
            bloq
            lowest_std_quantity

    Returns
    qpcr_data: same data, with additional column
    gc_per_L
    '''
    if replace_bloq:
        qpcr_data.loc[qpcr_data.bloq==True, "Quantity_mean"]= qpcr_data.lowest_std_quantity

    # calculate the conc of input to qPCR as gc/ul
    qpcr_data['gc_per_ul_input'] = qpcr_data['Quantity_mean'].astype(float) / qpcr_data['template_volume'].astype(float)

    # multiply input conc (gc / ul) by elution volume (ul) and divide by volume concentrated (mL). Multiply by 1000 to get to gc / L.
    qpcr_data['gc_per_L'] = 1000 * qpcr_data['gc_per_ul_input'].astype(float) * qpcr_data['elution_vol_ul'].astype(float) / qpcr_data['weight_vol_extracted_ml'].astype(float)



    return qpcr_data['gc_per_L']


def normalize_to_pmmov(qpcr_data, replace_bloq= False):

    '''
    calculates a normalized mean to pmmov when applicable and returns dataframe with new columns

      Params
        replace_bloq should the function replace any sample that is below the limit of quantification with the loq
        values the values to replace the bloq values with (should be lower than the loq)
        qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
                Target
                Quantity_mean
                Sample
                Task
            if replace_bloq is true
                bloq
                lowest_std_quantity
      Returns
      qpcr_m: same data, with additional columns
            mean_normalized_to_pmmov: takes every column and divides by PMMoV that is associated with that sample name (so where target == PMMoV it will be 1)
            log10mean_normalized_to_log10pmmov: takes the log10 of N1 and the log 10 of PMMoV then normalizes
            log10_mean_normalized_to_pmmov: takes the log10 of mean_normalized_to_pmmov
    '''
    if replace_bloq:
        qpcr_data.loc[qpcr_data.bloq==True, "Quantity_mean"]= qpcr_data.lowest_std_quantity

    pmmov=qpcr_data[qpcr_data.Target=='PMMoV']
    pmmov=pmmov[['Quantity_mean','Sample','Task']]
    pmmov.columns=['pmmov_mean',  "Sample", "Task"]
    qpcr_m=qpcr_data.merge(pmmov, how='left', on=["Sample", "Task"])
    qpcr_m["mean_normalized_to_pmmov"] = qpcr_m['Quantity_mean']/qpcr_m['pmmov_mean']
    qpcr_m["log10mean_normalized_to_log10pmmov"] = np.log10(qpcr_m['Quantity_mean'])/np.log10(qpcr_m['pmmov_mean'])
    qpcr_m['log10_mean_normalized_to_pmmov']=np.log10(qpcr_m['mean_normalized_to_pmmov'])

    return qpcr_m

def normalize_to_18S(qpcr_data, replace_bloq= False):

    '''
        calculates a normalized mean to 18S when applicable and returns dataframe with new columns

          Params
        replace_bloq should the function replace any sample that is below the limit of quantification with the loq
        values the values to replace the bloq values with (should be lower than the loq)
            qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
                    Target
                    Quantity_mean
                    Sample
                    Task
            if replace_bloq is true
                bloq
                lowest_std_quantity
          Returns
          qpcr_m: same data, with additional columns
                mean_normalized_to_18S: takes every column and divides by 18S that is associated with that sample name (so where target == 18S it will be 1)
                log10mean_normalized_to_log1018S: takes the log10 of N1 and the log 10 of 18S then normalizes
                log10_mean_normalized_to_18S: takes the log10 of mean_normalized_to_18S
    '''

    if replace_bloq:
        qpcr_data.loc[qpcr_data.bloq==True, "Quantity_mean"]= qpcr_data.lowest_std_quantity

    n_18S=qpcr_data[qpcr_data.Target=='18S']
    n_18S=n_18S[['Quantity_mean','Sample','Task']]
    n_18S.columns=['18S_mean',  "Sample", "Task"]
    qpcr_m=qpcr_data.merge(n_18S, how='left', on=["Sample", "Task"])
    qpcr_m["mean_normalized_to_18S"] = qpcr_m['Quantity_mean']/qpcr_m['18S_mean']
    qpcr_m["log10mean_normalized_to_log1018S"] = np.log10(qpcr_m['Quantity_mean'])/np.log10(qpcr_m['18S_mean'])
    qpcr_m['log10_mean_normalized_to_18S']=np.log10(qpcr_m['mean_normalized_to_18S'])


    return qpcr_m

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
        xeno_ntc_Cq_mean = sci.gmean(xeno_ntc_Cq_list)

    qpcr_processed_dilutions_xeno['xeno_dCt'] = qpcr_processed_dilutions_xeno.Cq_mean - xeno_ntc_Cq_mean
    qpcr_processed_dilutions_xeno['is_inhibited'] = None
    qpcr_processed_dilutions_xeno.loc[qpcr_processed_dilutions_xeno.xeno_dCt >= dCt_cutoff, 'is_inhibited'] = True
    qpcr_processed_dilutions_xeno.loc[qpcr_processed_dilutions_xeno.xeno_dCt < dCt_cutoff, 'is_inhibited'] = False

    qpcr_processed_dilutions_inhibition = qpcr_processed_dilutions_xeno[['Sample', 'dilution', 'xeno_dCt', 'is_inhibited', 'plate_id']]

    return(qpcr_processed_dilutions_inhibition)

def get_GFP_recovery(qpcr_averaged):
    '''
    calculate the percent recovery efficiency using GFP RNA spike
    Params
    qpcr_averaged: output of process_qpcr_raw(), a pandas df with columns
        GFP_spike_vol_ul
        Quantity_mean
        template_volume
        GFP_spike_tube
    Returns
    qpcr_averaged: the same df as input but with additional column
        perc_GFP_recovered
    '''
    gfp = qpcr_averaged[qpcr_averaged.Target == 'GFP'].copy()
    # calculate total recovered GFP gene copies
    gfp['total_GFP_recovered'] = gfp['GFP_spike_vol_ul'].astype(float) * gfp['Quantity_mean'].astype(float) / gfp['template_volume'].astype(float)

    # calculate concentration GFP gene copies / ul in just the spikes
    spikes = gfp[gfp.Sample.str.contains('control_spike_GFP')].copy()
    spikes['GFP_gc_per_ul_input'] = spikes['Quantity_mean'].astype(float) / spikes['template_volume'].astype(float)
    spikes = spikes[['GFP_gc_per_ul_input', 'GFP_spike_tube']]

    # combine concentration of spike and total recovered to get perc recovered
    gfp = gfp.merge(spikes, how = 'left', on = 'GFP_spike_tube')
    gfp['total_GFP_input'] = gfp['GFP_gc_per_ul_input'].astype(float) * gfp['GFP_spike_vol_ul'].astype(float)
    gfp['perc_GFP_recovered'] = 100 * gfp['total_GFP_recovered'] / gfp['total_GFP_input']
    recovery = gfp[['Sample', 'perc_GFP_recovered']]
    qpcr_averaged = qpcr_averaged.merge(recovery, how = 'left', on = 'Sample')
    return(qpcr_averaged)
