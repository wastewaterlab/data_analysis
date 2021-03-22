import pandas as pd
import requests


def preprocess_site_data(df_sites):
    '''rename fields from LabCollector db and subset fields
    to mesh with existing code'''
    df_sites = df_sites.rename(columns={'utility_code': 'utility', 'sample_code_concat': 'sample_code'})
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


def preprocess_sample_data(df_samples):
    '''rename fields from LabCollector db and subset fields
    to mesh with existing code'''
    df_samples = df_samples.rename(columns={'bcov_spike_tube': 'bCoV_spike_tube',
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
    df_samples.date_sampling = pd.to_datetime(df_samples.date_sampling, errors='coerce')
    df_samples.date_extract = pd.to_datetime(df_samples.date_extract, errors='coerce')
    df_samples.date_frozen = pd.to_datetime(df_samples.date_frozen, errors='coerce')
    df_samples.weight = pd.to_numeric(df_samples.weight, errors='coerce')
    df_samples.elution_vol_ul = pd.to_numeric(df_samples.elution_vol_ul, errors='coerce')
    df_samples.bCoV_spike_vol_ul = pd.to_numeric(df_samples.bCoV_spike_vol_ul, errors='coerce')
    df_samples.GFP_spike_vol_ul = pd.to_numeric(df_samples.GFP_spike_vol_ul, errors='coerce')

    return df_samples


def preprocess_plate_data(df_plates):
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

    return df_plates


def extract_dilution(qpcr_data):
    '''
    split the sample name if it starts with digitX_ (also handles lowercase x)
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


def preprocess_qpcr_data(qpcr_data, show_all_values=False):
    '''rename fields from LabCollector db and subset fields
    to mesh with existing code
    create additional fields and convert dtypes as needed by downstream code
    '''
    qpcr_data = qpcr_data[['omit', 'sample', 'target', 'dye', 'task', 'cq', 'quantity', 'well_id', 'plate_id', 'is_primary_value']].copy()

    qpcr_data = qpcr_data.rename(columns = {'omit': 'Omit',
                                    'sample': 'Sample',
                                    'target': 'Target',
                                    'dye': 'Dye',
                                    'task': 'Task',
                                    'cq': 'Cq',
                                    'quantity': 'Quantity'
                                   })

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
    qpcr_data['Target'] = qpcr_data['Target_full'].str.split(n=1, expand=True)[0]

    return qpcr_data


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
    df_samples = preprocess_sample_data(df_samples)
    df_plates = preprocess_plate_data(df_plates)
    df_qpcr = preprocess_qpcr_data(df_qpcr)

    return df_sites, df_samples, df_plates, df_qpcr
