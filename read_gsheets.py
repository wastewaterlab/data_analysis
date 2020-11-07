import pandas as pd
import numpy as np
import gspread
import warnings

def read_gsheet(gc, url, tab):
  '''Reads one tab from any google sheet,
  makes first row headers,
  replaces NA values,
  returns pandas dataframe.'''
  df = pd.DataFrame(gc.open_by_url(url).worksheet(tab).get_all_values())
  df.columns = df.iloc[0] #make first row the header
  df = df.iloc[1:]
  df=df.replace('NA',np.nan)
  return df

def read_sample_data(gc, samples_url, rna_tab, facility_lookup):
  ''' Reads in sample extraction and sample tracking data.
  Returns a dataframe of both merged'''

  rna_data = read_gsheet(gc, samples_url, rna_tab)
  facility_data = read_gsheet(gc, samples_url, facility_lookup)

  # left merge means keep Sample_extraction dataframe shape/rows and add sample inventory data to that
  #rna_data['utility'] = rna_data.sample_code.str.split('_', expand=True)[0]
  #rna_data['interceptor'] = rna_data.sample_code.str.split('_', n=1, expand=True)[1]
  rna_data = rna_data.merge(facility_data, how='left', on = 'sample_code')

  # convert fields to datetime and numeric
  rna_data['date_sampling'] = pd.to_datetime(rna_data['date_sampling'], errors='coerce') #convert date column to datetime #,errors='coerce'
  rna_data.elution_vol_ul = pd.to_numeric(rna_data.elution_vol_ul, errors='coerce')
  rna_data.effective_vol_extracted_ml = pd.to_numeric(rna_data.effective_vol_extracted_ml, errors='coerce')
  rna_data.bCoV_spike_vol_ul = pd.to_numeric(rna_data.bCoV_spike_vol_ul, errors='coerce')

  #check for duplicates
  a=rna_data[(rna_data.sample_id!="__")&(rna_data.sample_id!="")]
  a=a[a.duplicated(["sample_id"],keep=False)].copy()
  if len(a) > 0:
      samps=a.sample_id.unique()
      l=len(samps)
      warnings.warn("\n\n\n {0} samples are double listed in sample tracking spreadsheet. Check the following samples:\n\n\n{1}\n\n\n".format(l,samps))

  return rna_data


def adjust_for_dilution(qpcr_data):
      '''
      calculates adjusted quantity based on dilution,
      labels is_primary_value as "No" if there are multiple dilutions for the same sample,
      produces a dataframe that includes all the dilutions for a sample (if there are multiple)
      '''
      #subset
      qpcr_data['dilution']=1
      qpcr_data.loc[(qpcr_data.is_dilution== "Y"),"dilution"]=qpcr_data['Sample'].apply(lambda x: x.split('_')[0].replace('x',''))
      qpcr_data["Sample_full"]=qpcr_data["Sample"]
      # qpcr_inhibition["Sample"]=qpcr_inhibition["Sample"].apply(lambda x: x.split('_',1)[1])
      return(qpcr_data)

def read_qpcr_data(gc, qpcr_url, qpcr_results_tab, qpcr_plates_tab):
  ''' Read in raw qPCR data page from the qPCR spreadsheet
  '''
  qpcr_data = read_gsheet(gc, qpcr_url, qpcr_results_tab)
  qpcr_plates = read_gsheet(gc, qpcr_url, qpcr_plates_tab)
  qpcr_data = qpcr_data.merge(qpcr_plates, how='left', on='plate_id')
  qpcr_data =adjust_for_dilution(qpcr_data)


  # filter to remove secondary values for a sample run more than once
  qpcr_data=qpcr_data[qpcr_data.is_primary_value=='Y']

  # create field for sample-plate combos in case same sample run on >1 plate
  qpcr_data['Sample_plate']= qpcr_data.Sample.str.cat(qpcr_data.plate_id, sep ="+")

  # create column to preserve info about undetermined values
  qpcr_data['is_undetermined'] = False
  qpcr_data['is_undetermined'] = qpcr_data.Cq == 'Undetermined'

  # convert fields to numerics and dates
  qpcr_data.Quantity = pd.to_numeric(qpcr_data.Quantity, errors='coerce')
  qpcr_data.template_volume = pd.to_numeric(qpcr_data.template_volume, errors='coerce')
  qpcr_data.Cq = pd.to_numeric(qpcr_data.Cq, errors='coerce')
  qpcr_data.plate_id = pd.to_numeric(qpcr_data.plate_id, errors='coerce')
  qpcr_data.Plate_date = pd.to_datetime(qpcr_data.Plate_date, errors='coerce')

  # get a column with only the target (separate info about the standard and master mix)
  qpcr_data['Target_full']= qpcr_data['Target']
  qpcr_data['Target']=qpcr_data['Target'].apply(lambda x: x.split()[0])

  return(qpcr_data)
