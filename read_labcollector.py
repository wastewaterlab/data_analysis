import pandas as pd
import requests

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

    return df_sites, df_samples, df_plates, df_qpcr


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


def preprocess_qpcr_from_LabCollector_df(qpcr_data, show_all_values=False):

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
