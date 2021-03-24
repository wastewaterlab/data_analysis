import pandas as pd
import numpy as np
import gspread
import re
import warnings

#TODO delete effective_vol_extracted_ml column from data tables, just use weight
def read_table(gc, url, tab):
    '''
    Reads pandas df or one tab from any google sheet,
    makes first row headers,
    replaces NA values,
    returns pandas dataframe
    '''
    if gc is None:
        df = pd.read_csv(tab)
    else:
        df = pd.DataFrame(gc.open_by_url(url).worksheet(tab).get_all_values())
        df.columns = df.iloc[0] #make first row the header
        df = df.iloc[1:]
        df=df.replace('NA',np.nan)
    return df


def read_sample_data(gc, url, samples, sites, salted_tube_weight=23.485):
    '''
    merge the sample df and sites df, convert fields to correct dtypes, calculate the volume extracted based on sample weight

    Params
    gc: google credentials or None if reading from files
    url: google sheet url or None if reading from files
    samples: csv or tab name for samples table
    sites_df: csv or tab name for sites table
    salted_tube_weight: experimentally determined average weight of tube with the salt preservative prior to sample addition

    Returns
    samples_sites_df: pandas dataframe where each row is a unique sample with site info
    '''
    # default values, not integrated into the code, but could be (i.e., if the value in the column is NaN, make it the default)
    defaults = {'elution_vol_ul': 200, 'effective_vol_extracted_ml': 40, 'weight_vol_extracted_ml':40, 'bCoV_spike_vol_ul': 50, 'GFP_spike_vol_ul': 20}

    samples_df = read_table(gc, url, samples)
    sites_df = read_table(gc, url, sites)

    # remove empty rows
    samples_df = samples_df[~samples_df.sample_code.isna()]

    # convert fields to datetime and numeric
    samples_df.date_sampling = pd.to_datetime(samples_df.date_sampling, errors='coerce')
    samples_df.date_extract = pd.to_datetime(samples_df.date_extract, errors='coerce')
    samples_df.elution_vol_ul = pd.to_numeric(samples_df.elution_vol_ul, errors='coerce')
    samples_df.weight = pd.to_numeric(samples_df.weight, errors='coerce')
    samples_df.bCoV_spike_vol_ul = pd.to_numeric(samples_df.bCoV_spike_vol_ul, errors='coerce')
    samples_df.GFP_spike_vol_ul = pd.to_numeric(samples_df.GFP_spike_vol_ul, errors='coerce')

    # instead of assuming all samples are 40 mL, use the weight of the sample
    # minus weight of the tube, which we experimentally measured 10 times
    samples_df['weight_vol_extracted_ml'] = samples_df.weight - salted_tube_weight
    # if weight is missing, assign value of 40
    samples_df.loc[samples_df.weight_vol_extracted_ml.isna(), 'weight_vol_extracted_ml'] = 40

    # check for duplicates
    if not len(samples_df.sample_id) == len(set(samples_df.sample_id)):
        duplicates = samples_df[samples_df.sample_id.duplicated()].sample_id.to_list()
        warnings.warn(f'duplicate sample_id: {duplicates}')

    samples_sites_df = samples_df.merge(sites_df, how='left', on = 'sample_code')

    if not len(samples_df) == len(samples_sites_df):
        warnings.warn("merging samples and sites introduced new rows")

    return samples_sites_df


def extract_dilution(qpcr_data):
    '''
    splits the sample name if it starts with digitX_ (also handles lowercase x)
    captures the dilution in a new column, renames samples
    if samples were not diluted, dilution will be 1
    '''
    dilution_sample_names_df = qpcr_data.Sample.str.extract(r'^(\d+)[X,x]_(.+)', expand=True)
    dilution_sample_names_df = dilution_sample_names_df.rename(columns = {0: 'dilution', 1: 'Sample_new'})
    qpcr_data = pd.concat([qpcr_data, dilution_sample_names_df], axis=1)
    qpcr_data.loc[qpcr_data.Sample_new.isna(), 'Sample_new'] = qpcr_data.Sample
    qpcr_data.loc[qpcr_data.dilution.isna(), 'dilution'] = 1
    qpcr_data = qpcr_data.rename(columns = {'Sample_new' : 'Sample', 'Sample' : 'sample_full'})
    qpcr_data.dilution = pd.to_numeric(qpcr_data.dilution)

    return qpcr_data

def read_qpcr_data(gc, url, qpcr, show_all_values=False):
  ''' Read in raw qPCR data page from the qPCR spreadsheet
  '''
  qpcr_data = read_table(gc, url, qpcr)
  qpcr_data = extract_dilution(qpcr_data)

  # filter to remove secondary values for a sample run more than once
  if show_all_values is False:
      qpcr_data = qpcr_data[qpcr_data.is_primary_value != 'N']

  # create column to preserve info about true undetermined values
  # set column equal to boolean outcome of asking if Cq is Undetermined
  qpcr_data['is_undetermined'] = False
  qpcr_data['is_undetermined'] = (qpcr_data.Cq == 'Undetermined')

  # convert fields to numerics and dates
  qpcr_data.Quantity = pd.to_numeric(qpcr_data.Quantity, errors='coerce')
  qpcr_data.Cq = pd.to_numeric(qpcr_data.Cq, errors='coerce')
  qpcr_data.plate_id = pd.to_numeric(qpcr_data.plate_id, errors='coerce')

  # get a column with only the target (separate info about the standard and master mix)
  qpcr_data['Target_full'] = qpcr_data['Target']
  qpcr_data['Target'] = qpcr_data['Target'].apply(lambda x: x.split()[0])

  return qpcr_data
