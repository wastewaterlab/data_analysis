import pandas as pd
import numpy as np
from scipy import stats as sci
from scipy.stats.mstats import gmean
from scipy.stats import gstd
import warnings

def calculate_gc_per_ml(qpcr_data):
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
    qpcr_data['Quantity_mean'] = qpcr_data['Quantity_mean'].astype(float)
    qpcr_data['template_volume'] = qpcr_data['template_volume'].astype(float)
    qpcr_data['elution_vol_ul'] = qpcr_data['elution_vol_ul'].astype(float)
    qpcr_data['weight_vol_extracted_ml'] = qpcr_data['weight_vol_extracted_ml'].astype(float)

    # calculate the conc of input to qPCR as gc/ul
    qpcr_data['gc_per_ul_input'] = qpcr_data['Quantity_mean'] / qpcr_data['template_volume']

    # multiply input conc (gc / ul) by elution volume (ul) and divide by volume concentrated (mL).
    qpcr_data['gc_per_mL'] = qpcr_data['gc_per_ul_input'] * qpcr_data['elution_vol_ul'] / qpcr_data['weight_vol_extracted_ml']

    return(qpcr_data['gc_per_mL'])


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

    df['elution_vol_ul'] = df['elution_vol_ul'].astype(float)
    df['Quantity_mean'] = df['Quantity_mean'].astype(float)
    df['template_volume'] = df['template_volume'].astype(float)
    df[f'{spike_name}_spike_vol_ul'] = df[f'{spike_name}_spike_vol_ul'].astype(float)

    # calculate total recovered gene copies
    df['total_recovered'] = df['elution_vol_ul'] * df['Quantity_mean'] / df['template_volume']

    # calculate concentration gene copies / ul in just the spikes
    spikes = df[df.Sample.str.contains(f'control_spike_{spike_name}')].copy()
    spikes[f'{spike_name}_gc_per_ul_input'] = spikes['Quantity_mean'] / spikes['template_volume']

    # fill in missing spike tube names
    spikes[f'{spike_name}_spike_tube'] = spikes.Sample.str.rsplit('_', n=1, expand=True)[1]
    spikes = spikes[[f'{spike_name}_gc_per_ul_input', f'{spike_name}_spike_tube']]
    if len(spikes[spikes[f'{spike_name}_spike_tube'].isna()]) > 0:
        warnings.warn('some spike tubes are unnamed')

    # # combine concentration of spike and total recovered to get perc recovered
    df = df.merge(spikes, how = 'left', on = f'{spike_name}_spike_tube')
    df['total_input'] = df[f'{spike_name}_gc_per_ul_input'] * df[f'{spike_name}_spike_vol_ul']
    df[f'{spike_name}_perc_recovered'] = 100 * df['total_recovered'] / df['total_input']
    recovery_df = df[['Sample', f'{spike_name}_perc_recovered', f'{spike_name}_gc_per_ul_input']]

    # drop rows that are the spikes
    recovery_df = recovery_df[~recovery_df.Sample.str.contains(f'control_spike_{spike_name}')]
    # drop rows where recovery was NaN due to spike not amplifying
    recovery_df = recovery_df[~recovery_df[f'{spike_name}_perc_recovered'].isna()]

    return(recovery_df)
