import pandas as pd
import numpy as np
from scipy import stats as sci
from scipy.stats.mstats import gmean
from scipy.stats import gstd

def calculate_gc_per_ml(qpcr_data, replace_bloq=False ):
    '''
    calculates and returns gene copies / mL

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
    gc_per_mL
    '''
    if replace_bloq:
        qpcr_data.loc[qpcr_data.bloq==True, "Quantity_mean"]= qpcr_data.lowest_std_quantity

    # calculate the conc of input to qPCR as gc/ul
    qpcr_data['gc_per_ul_input'] = qpcr_data['Quantity_mean'].astype(float) / qpcr_data['template_volume'].astype(float)

    # multiply input conc (gc / ul) by elution volume (ul) and divide by volume concentrated (mL).
    qpcr_data['gc_per_mL'] = qpcr_data['gc_per_ul_input'].astype(float) * qpcr_data['elution_vol_ul'].astype(float) / qpcr_data['weight_vol_extracted_ml'].astype(float)

    return(qpcr_data['gc_per_mL'])


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
    pmmov=pmmov[['Quantity_mean','Sample']]
    pmmov.columns=['pmmov_mean',  "Sample"]
    qpcr_m=qpcr_data.merge(pmmov, how='left', on=["Sample"])
    qpcr_m["mean_normalized_to_pmmov"] = qpcr_m['Quantity_mean']/qpcr_m['pmmov_mean']
    qpcr_m["log10mean_normalized_to_log10pmmov"] = np.log10(qpcr_m['Quantity_mean'])/np.log10(qpcr_m['pmmov_mean'])
    qpcr_m['log10_mean_normalized_to_pmmov']=np.log10(qpcr_m['mean_normalized_to_pmmov'])

    return qpcr_m


def calculate_recovery(sample_data_qpcr, spike_name):
    '''
    Given the sample_data_qpcr dataframe,
    calculate the total amount of the spike that was recovered in samples
    calculate total amount of the spike that was added to samples
    calculate percent recovery as output spike amount / input spike amount

    Params
    sample_data_qpcr: dataframe containing sample inventory and qPCR data
    including rows for spikes and columns:
        f'{spike_name}_spike_tube': contains name of the spike aliquot tube
        f'{spike_name}_spike_vol_ul': contains volume of spike added to sample

    Returns
    recovery_df: dataframe with same Sample names with rows containing
    spike tubes removed. Columns:
        'Sample': for merging back to the original dataframe
         f'{spike_name}_perc_recovered': percent recovery of the spike
         f'{spike_name}_gc_per_ul_input': input concentration of the spike
    '''

    if spike_name not in ['bCoV', 'GFP']:
        raise ValueError('the spike name must be bCoV or GFP')

    df = sample_data_qpcr[sample_data_qpcr.Target == spike_name].copy()
    # calculate total recovered gene copies
    df['total_recovered'] = df['elution_vol_ul'].astype(float) * df['Quantity_mean'].astype(float) / df['template_volume'].astype(float)

    # calculate concentration gene copies / ul in just the spikes
    spikes = df[df.Sample.str.contains(f'control_spike_{spike_name}')].copy()
    spikes[f'{spike_name}_gc_per_ul_input'] = spikes['Quantity_mean'].astype(float) / spikes['template_volume'].astype(float)
    spikes = spikes[[f'{spike_name}_gc_per_ul_input', f'{spike_name}_spike_tube']]

    # # combine concentration of spike and total recovered to get perc recovered
    df = df.merge(spikes, how = 'left', on = f'{spike_name}_spike_tube')
    df['total_input'] = df[f'{spike_name}_gc_per_ul_input'].astype(float) * df[f'{spike_name}_spike_vol_ul'].astype(float)
    df[f'{spike_name}_perc_recovered'] = 100 * df['total_recovered'] / df['total_input']
    recovery_df = df[['Sample', f'{spike_name}_perc_recovered', f'{spike_name}_gc_per_ul_input']]
    recovery_df = recovery_df[~recovery_df.Sample.str.contains(f'control_spike_{spike_name}')] # drop rows that are the spikes
    return(recovery_df)
