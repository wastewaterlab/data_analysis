import pandas as pd
import numpy as np

def calculate_gc_per_l(qpcr_data):
 '''
 calculates and returns gene copies / L

  Params
    qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
            gc_per_ul_input
            Quantity_mean
            template_volume
            elution_vol_ul
            effective_vol_extracted_ml

  Returns
  qpcr_data: same data, with additional column
  gc_per_L
  '''

  # calculate the conc of input to qPCR as gc/ul
  qpcr_data['gc_per_ul_input'] = qpcr_data['Quantity_mean'].astype(float) / qpcr_data['template_volume'].astype(float)

  ## multiply input conc (gc / ul) by elution volume (ul) and divide by volume concentrated (mL). Multiply by 1000 to get to gc / L.
  qpcr_data['gc_per_L'] = 1000 * qpcr_data['gc_per_ul_input'].astype(float) * qpcr_data['elution_vol_ul'].astype(float) / qpcr_data['effective_vol_extracted_ml'].astype(float)
  return qpcr_data['gc_per_L']


def normalize_to_pmmov(qpcr_data):

    '''
    calculates a normalized mean to pmmov when applicable and returns dataframe with new columns

      Params
        qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
                Target
                Quantity_mean
                Sample
                Task
      Returns
      qpcr_m: same data, with additional columns
            mean_normalized_to_pmmov: takes every column and divides by PMMoV that is associated with that sample name (so where target == PMMoV it will be 1)
            log10mean_normalized_to_log10pmmov: takes the log10 of N1 and the log 10 of PMMoV then normalizes
            log10_mean_normalized_to_pmmov: takes the log10 of mean_normalized_to_pmmov
    '''
    pmmov=qpcr_data[qpcr_data.Target=='PMMoV']
    pmmov=pmmov[['Quantity_mean','Sample','Task']]
    pmmov.columns=['pmmov_mean',  "Sample", "Task"]
    qpcr_m=qpcr_data.merge(pmmov, how='left')
    qpcr_m["mean_normalized_to_pmmov"] = qpcr_m['Quantity_mean']/qpcr_m['pmmov_mean']
    qpcr_m["log10mean_normalized_to_log10pmmov"] = np.log10(qpcr_m['Quantity_mean'])/np.log10(qpcr_m['pmmov_mean'])
    qpcr_m['log10_mean_normalized_to_pmmov']=np.log10(qpcr_m['mean_normalized_to_pmmov'])

    return qpcr_m

def normalize_to_18S(qpcr_data):

    '''
        calculates a normalized mean to 18S when applicable and returns dataframe with new columns

          Params
            qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
                    Target
                    Quantity_mean
                    Sample
                    Task
          Returns
          qpcr_m: same data, with additional columns
                mean_normalized_to_18S: takes every column and divides by 18S that is associated with that sample name (so where target == 18S it will be 1)
                log10mean_normalized_to_log1018S: takes the log10 of N1 and the log 10 of 18S then normalizes
                log10_mean_normalized_to_18S: takes the log10 of mean_normalized_to_18S
    '''

    n_18S=qpcr_data[qpcr_data.Target=='18S']
    n_18S=n_18S[['Quantity_mean','Sample','Task']]
    n_18S.columns=['18S_mean',  "Sample", "Task"]
    qpcr_m=qpcr_data.merge(n_18S, how='left')
    qpcr_m["mean_normalized_to_18S"] = qpcr_m['Quantity_mean']/qpcr_m['18S_mean']
    qpcr_m["log10mean_normalized_to_log1018S"] = np.log10(qpcr_m['Quantity_mean'])/np.log10(qpcr_m['18S_mean'])
    qpcr_m['log10_mean_normalized_to_18S']=np.log10(qpcr_m['mean_normalized_to_18S'])

    return qpcr_m

def xeno_inhibition_test(qpcr_data):
  '''
        Calculates the difference in Ct compared to the NTC for xeno inhibition test, outputs a list of inhibited samples

          Params
            qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
                    Target
                    plate_id
                    Well
                    Quantity_mean
                    Sample
                    Task
          Returns
          xeno_fin_all -- calculates the difference in Ct values of the negative control (spiked with xeno) to the sample spiked with xeno, adds column for inhibited (Yes or No)
          ntc_col -- all of the negative control values for xeno
  '''

  #Find targets other than xeno for each well+plate combination
  p_w_targets=qpcr_data[qpcr_data.Target!='Xeno'].copy()
  p_w_targets['p_id']=p_w_targets.plate_id.astype('str').str.cat(p_w_targets.Well.astype('str'), sep ="_")
  p_w_targets=p_w_targets.groupby('p_id')['Target'].apply(lambda targs: ','.join(targs)).reset_index()
  p_w_targets.columns=['p_id','additional_target']

  #subset out xeno samples, merge with previous, use to calculate mean and std
  target=qpcr_data[(qpcr_data.Task!='Standard')&(qpcr_data.Target=='Xeno')].copy() #includes NTC
  target['p_id']=qpcr_data.plate_id.astype('str').str.cat(qpcr_data.Well, sep ="_")
  target=target.merge(p_w_targets, how='left', on='p_id')
  if target.additional_target.astype('str').str.contains(',').any():
      print(target[target.additional_target.str.contains(',')])
      raise ValueError('Error: update function, more than 2 multiplexed targets or one of the two multiplexed targets is not xeno')

  target=target.groupby(["Sample",'additional_target','plate_id','Task']).agg(Ct_vet_mean=('Cq', 'mean'),
                                                                    Ct_vet_std=('Cq', 'std'),
                                                                    Ct_vet_count=('Cq','count')).reset_index()
  #subset and recombine to get NTC as a col
  ntc_col=target[target.Task=='Negative Control'].copy()
  ntc_col=ntc_col[["plate_id",'additional_target','Ct_vet_mean']].copy()
  ntc_col.columns=["plate_id",'additional_target','Ct_control_mean']

  xeno_fin_all=target[target.Task=='Unknown'].copy()
  xeno_fin_all=xeno_fin_all.merge(ntc_col, how='left')
  xeno_fin_all["dCt"]= (xeno_fin_all["Ct_vet_mean"]- xeno_fin_all["Ct_control_mean"])
  xeno_fin_all["inhibited"]='No'
  xeno_fin_all.loc[(xeno_fin_all.dCt>1),"inhibited"]="Yes"
  return xeno_fin_all, ntc_col

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
