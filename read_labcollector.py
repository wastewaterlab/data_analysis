import numpy as np
import pandas as pd
import requests


def preprocess_site_data(df_sites):
    '''rename fields from LabCollector db and subset fields
    to mesh with existing code'''
    df_sites = df_sites.rename(columns={'utility_code': 'utility', 'name': 'sample_code'})
    df_sites = df_sites[['utility_name',
                       'county',
                       'utility',
                       'facility',
                       'location',
                       'sample_code',
                       'sample_level',
                       'sampling_days',
                       'sampling_frequency_per_week',
                       'site_full_name',
                       'site_description',
                       'site_pretreatment',
                       'site_population_served'
                      ]].copy()
    return df_sites


def preprocess_sample_data(df_samples, df_sites, salted_tube_weight=23.485):
    '''rename fields from LabCollector db and subset fields
    to mesh with existing code'''
    df_samples = df_samples.drop(columns = 'sample_id') # this column is empty, should use column "label"
    df_samples = df_samples.rename(columns={'label': 'sample_id',
                                            'bcov_spike_tube': 'bCoV_spike_tube',
                                            'gfp_spike_tube': 'GFP_spike_tube',
                                            'qpcr_batch': 'qPCR_batch',
                                            'bcov_spike_vol_ul': 'bCoV_spike_vol_ul',
                                            'gfp_spike_vol_ul': 'GFP_spike_vol_ul'
                                           })
    df_samples = df_samples[[
                             'sample_id',
                             'batch',
                             'sample_code',
                             'date_sampling',
                             'replicate',
                             'date_extract',
                             'extracted_by',
                             'weight',
                             'bCoV_spike_tube',
                             'GFP_spike_tube',
                             'processing_error',
                             'qPCR_batch',
                             'storage_conditions',
                             'salted_before_freezing',
                             'date_frozen',
                             'elution_vol_ul',
                             'bCoV_spike_vol_ul',
                             'GFP_spike_vol_ul',
                            ]].copy()
    # drop rows without sample_code (means row is empty)
    df_samples = df_samples[~(df_samples.sample_code.isna())]

    df_samples.date_sampling = pd.to_datetime(df_samples.date_sampling, errors='coerce')
    df_samples.date_extract = pd.to_datetime(df_samples.date_extract, errors='coerce')
    df_samples.date_frozen = pd.to_datetime(df_samples.date_frozen, errors='coerce')
    df_samples.weight = pd.to_numeric(df_samples.weight, errors='coerce')
    df_samples.elution_vol_ul = pd.to_numeric(df_samples.elution_vol_ul, errors='coerce')
    df_samples.bCoV_spike_vol_ul = pd.to_numeric(df_samples.bCoV_spike_vol_ul, errors='coerce')
    df_samples.GFP_spike_vol_ul = pd.to_numeric(df_samples.GFP_spike_vol_ul, errors='coerce')

    # substitute NaN for empty string so pd.isnull() will run on this field
    df_samples.loc[df_samples.processing_error == '', 'processing_error'] = np.nan

    # instead of assuming all samples are 40 mL, use the weight of the sample
    # minus weight of the tube, which we experimentally measured 10 times
    df_samples['weight_vol_extracted_ml'] = df_samples.weight - salted_tube_weight
    # if weight is missing, assign value of 40
    df_samples.loc[df_samples.weight_vol_extracted_ml.isna(), 'weight_vol_extracted_ml'] = 40

    if not len(df_samples.sample_id) == len(set(df_samples.sample_id)):
        duplicates = df_samples[df_samples.sample_id.duplicated()].sample_id.to_list()
        warnings.warn(f'duplicate sample_id: {duplicates}')

    df_samples_sites = df_samples.merge(df_sites, how='left', on = 'sample_code')

    if not len(df_samples) == len(df_samples_sites):
        warnings.warn("merging samples and sites introduced new rows")

    return df_samples_sites


def preprocess_plate_data(df_plates, rxn_volume=20.0, template_volume=5.0):
    df_plates = df_plates[[
                         'plate_id',
                         'plate_file_name',
                         'plate_date',
                         'assays',
                         'standard_type',
                         'standard_batch',
                         'comment_log',
                         'run_by',
                         'rxn_volume',
                         'template_volume',
                         'plate_notes',
                         'standard_curve_by',
                         'plate_by'
                        ]].copy()
    df_plates.plate_id = pd.to_numeric(df_plates.plate_id, errors='coerce')
    df_plates.plate_date = pd.to_datetime(df_plates.plate_date, errors='coerce')
    df_plates.rxn_volume = pd.to_numeric(df_plates.rxn_volume, errors='coerce')
    df_plates.template_volume = pd.to_numeric(df_plates.template_volume, errors='coerce')

    df_plates[~df_plates.plate_id.isna()] # drop empty rows (must have plate_id)
    # fill in default values
    df_plates.loc[df_plates.rxn_volume.isna(), 'rxn_volume'] = rxn_volume
    df_plates.loc[df_plates.template_volume.isna(), 'template_volume'] = template_volume

    if not len(df_plates.plate_id) == len(set(df_plates.plate_id)):
        # check for duplicate plate_id
        raise ValueError('multiple plate info entries with same plate_id')

    return df_plates


def extract_dilution(df_qpcr):
    '''
    split the sample name if it starts with digitX_ (also handles lowercase x)
    captures the dilution in a new column, renames samples
    if samples were not diluted, dilution will be 1
    '''
    dilution_sample_names_df = df_qpcr.Sample.str.extract(r'^(\d+)[X,x]_(.+)', expand=True)
    dilution_sample_names_df = dilution_sample_names_df.rename(columns = {0: 'dilution', 1: 'Sample_new'})
    df_qpcr = pd.concat([df_qpcr, dilution_sample_names_df], axis=1)
    df_qpcr.loc[df_qpcr.Sample_new.isna(), 'Sample_new'] = df_qpcr.Sample
    df_qpcr.loc[df_qpcr.dilution.isna(), 'dilution'] = 1
    df_qpcr = df_qpcr.rename(columns = {'Sample_new' : 'Sample', 'Sample' : 'sample_full'})
    df_qpcr.dilution = pd.to_numeric(df_qpcr.dilution)

    return df_qpcr


def preprocess_df_qpcr(df_qpcr, show_all_values=False):
    '''rename fields from LabCollector db and subset fields
    to mesh with existing code
    create additional fields and convert dtypes as needed by downstream code
    '''
    df_qpcr = df_qpcr[['omit', 'sample', 'target', 'dye', 'task', 'cq', 'quantity', 'well_id', 'plate_id', 'is_primary_value']].copy()

    df_qpcr = df_qpcr.rename(columns = {'omit': 'Omit',
                                    'sample': 'Sample',
                                    'target': 'Target',
                                    'dye': 'Dye',
                                    'task': 'Task',
                                    'cq': 'Cq',
                                    'quantity': 'Quantity'
                                   })

    df_qpcr = extract_dilution(df_qpcr)

    # drop empty wells (these shouldn't be imported but can happen by accident)
    df_qpcr = df_qpcr[~df_qpcr.Sample.isna()]

    # filter to remove secondary values for a sample run more than once
    if show_all_values is False:
        df_qpcr = df_qpcr[df_qpcr.is_primary_value != 'N']

    # create column to preserve info about true undetermined values
    # set column equal to boolean outcome of asking if Cq is Undetermined
    df_qpcr['is_undetermined'] = False
    df_qpcr['is_undetermined'] = (df_qpcr.Cq == 'Undetermined')

    # convert fields to numerics and dates
    df_qpcr.Quantity = pd.to_numeric(df_qpcr.Quantity, errors='coerce')
    df_qpcr.Cq = pd.to_numeric(df_qpcr.Cq, errors='coerce')
    df_qpcr.plate_id = pd.to_numeric(df_qpcr.plate_id, errors='coerce')

    # get a column with only the target (separate info about the standard and master mix)
    df_qpcr['Target_full'] = df_qpcr['Target']
    df_qpcr['Target'] = df_qpcr['Target_full'].str.split(n=1, expand=True)[0]

    return df_qpcr


def make_api_call(url, token):
    headers = {
        'X-LC-APP-Auth': token,
        'Accept': 'application/json'
    }

    sites = "sample_locations"
    samples = "samples"
    plates = "qpcr_plate_info"
    qpcr = "qpcr_raw_data"

    r = requests.get(url+sites, headers=headers)
    df_sites = pd.read_json(r.text)
    r = requests.get(url+samples, headers=headers)
    df_samples = pd.read_json(r.text)
    r = requests.get(url+plates, headers=headers)
    df_plates = pd.read_json(r.text)
    r = requests.get(url+qpcr, headers=headers)
    df_qpcr = pd.read_json(r.text)

    # see other functions for preprocessing
    # note: df_plates doesn't need any renaming from LabCollector to mesh with existing code

    df_sites = preprocess_site_data(df_sites)
    df_samples_sites = preprocess_sample_data(df_samples, df_sites)
    df_plates = preprocess_plate_data(df_plates)
    df_qpcr = preprocess_df_qpcr(df_qpcr)

    return df_samples_sites, df_plates, df_qpcr
