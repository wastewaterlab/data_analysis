
import pandas as pd
import numpy  as np

def quality_score(p, dic_name, df):
   '''
   for a processed data frame, will give quality score info

   params
   p is a dictionary indicating point values. All point values have 3 conditions
   where C1 meets acceptable qa/qc for that column, C2 is slightly concerning, and C3 is
   very concerning. The weight should be between 0 and 1 and compares the dictionary entries to eachother
   p should be in the format of 'dic_name': [weight, C1, C2, C3]. The dic_name needs to be one/all/any of these strings:
   "eff_std", "Rsq_std", "n_std", "used_cong_std", "NTC_std", "n_reps","is_inhibited", "transit_days","PBS_amp"

   example: p={"eff_std":[0.06,1,0.7,0.2], "Rsq_std":[0.06,1,0.7,0.2], "n_std":[0.07,1,0.2,0],
   "used_cong_std":[0.04,1,0],"NTC_std":[0.1,1,0.8,0],"n_reps":[0.2,1,0.8,0],
   "is_inhibited":[0.1,1,0], "transit_days":[0.09,1,0.8,0], "PBS_amp":[0.1,1,0.8,0]}

   dic_name is a list of strings in p to include in the analysis

   example: ['eff_std', "Rsq_std",'n_std','n_reps','NTC_std','is_inhibited']

   df columns (should correspond to columns listed in dic_name):
   Target
   efficiency
   r2
   num_points
   ntc_result & Cq_of_lowest_std_quantity
   replicate_count
   is_inhibited
   date_conc_extract & date_sampling & stored_minus_80 & stored_minus_20
   PBS_result & Cq_of_lowest_std_quantity

   result:
   returns the same dataframe with the following columns

   quality_score: points from p as a percentage

   flags: any samples with quality_score = np.nan have
   flag(s) indicating why in the flags column

   point_deduction: provides reason(s) for point deduction

   '''
   df=df.copy()
   df['quality_score']= 0
   max_score=0
   df['flag']=""
   df['point_deduction']=""

   #calculate max score
   for name in dic_name:
       max_score= max_score+ p[name][0]* p[name][1]

   # efficiency of the standard curve
   e="eff_std"
   if e in dic_name:
        for row in df.itertuples():
            if (row.Target!= 'Xeno')&(~np.isnan(row.efficiency)):
                  if ((row.efficiency >=0.8) | (row.efficiency <=1.1)) :
                    value= row.quality_score + p[e][0]*p[e][1]
                    df.loc[row.Index,'quality_score'] = value
                  elif ((row.efficiency >=0.7) | (row.efficiency <=1.2)) :
                    value= row.quality_score + p[e][0]*p[e][2]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " efficiency (2);"
                  else:
                    value= row.quality_score  + p[e][0]*p[e][3]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " efficiency (3);"
            else:
              df.loc[row.Index,'quality_score'] = np.nan
              if np.isnan(row.efficiency):
                df.loc[row.Index,'flag'] = df.loc[row.Index,'flag'] + " check standard curve efficiency;"
   # R squared
   e="Rsq_std"
   if e in dic_name:
         for row in df.itertuples():
             if (row.Target!= 'Xeno')&(~np.isnan(row.r2)):
                  if (row.r2 >=0.98):
                    value= row.quality_score + p[e][0]*p[e][1]
                    df.loc[row.Index,'quality_score'] = value
                  elif (row.r2 >=0.9):
                    value= row.quality_score + p[e][0]*p[e][2]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " R2 (2);"
                  else:
                    value= row.quality_score  + p[e][0]*p[e][3]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " R2 (3);"
             else:
              df.loc[row.Index,'quality_score'] = np.nan
              if np.isnan(row.r2):
                df.loc[row.Index,'flag'] = df.loc[row.Index,'flag'] + " check standard curve R2;"

   # number of points in the std curve
   e="n_std"
   if e in dic_name:
         for row in df.itertuples():
             if (row.Target!= 'Xeno')&(~np.isnan(row.num_points)):
                  if (row.num_points >=5):
                    value= row.quality_score + p[e][0]*p[e][1]
                    df.loc[row.Index,'quality_score'] = value
                  elif (row.num_points >=3):
                    value= row.quality_score + p[e][0]*p[e][2]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " number of points in std curve (2);"
                  else:
                    value= row.quality_score  + p[e][0]*p[e][3]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " number of points in std curve (3);"
             else:
               df.loc[row.Index,'quality_score'] = np.nan
               if np.isnan(row.num_points):
                 df.loc[row.Index,'flag'] = df.loc[row.Index,'flag'] + " check standard curve number of points;"

   # number of replicates
   e="n_reps"
   if e in dic_name:
        for row in df.itertuples():
           if (row.Target!= 'Xeno')&(~np.isnan(row.replicate_count)):
                  if (row.replicate_count >=3):
                    value= row.quality_score + p[e][0]*p[e][1]
                    df.loc[row.Index,'quality_score'] = value
                  elif (row.replicate_count >=2):
                    value= row.quality_score + p[e][0]*p[e][2]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " number of replicates (2);"
                  else:
                    value= row.quality_score  + p[e][0]*p[e][3]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " number of replicates (3);"
                    if row.replicate_count==0:
                      df.loc[row.Index,'flag'] = 'set to 0'
                      df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " 0 replicates;"
           else:
              df.loc[row.Index,'quality_score'] = np.nan
              if np.isnan(row.replicate_count):
                df.loc[row.Index,'flag'] = df.loc[row.Index,'flag'] + " check sample number of replicates;"

   #NTC
   e="NTC_std"
   if e in dic_name:
        for row in df.itertuples():
           if (row.Target!= 'Xeno')&(row.ntc_result!= ""):
                  if (row.ntc_result =='negative'):
                    value= row.quality_score + p[e][0]*p[e][1]
                    df.loc[row.Index,'quality_score'] = value
                  elif (pd.to_numeric(row.ntc_result) < pd.to_numeric(row.Cq_of_lowest_std_quantity)+1):
                    value= row.quality_score + p[e][0]*p[e][2]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " NTC (2);"
                  else:
                    value= row.quality_score  + p[e][0]*p[e][3]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " NTC (3);"
           else:
              df.loc[row.Index,'quality_score'] = np.nan
              if (row.ntc_result== ""):
                df.loc[row.Index,'flag'] = df.loc[row.Index,'flag'] + " check standard curve NTCs;"

   #is_inhibited
   e="is_inhibited"
   if e in dic_name:
      for row in df.itertuples():
           if (row.Target!= 'Xeno')&(row.is_inhibited != "") &(row.is_inhibited != "unknown"):
                  if (row.is_inhibited =='No'):
                    value= row.quality_score + p[e][0]*p[e][1]
                    df.loc[row.Index,'quality_score'] = value
                  else:
                    value= row.quality_score  + p[e][0]*p[e][2]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " sample is inhibited;"
           else:
              df.loc[row.Index,'quality_score'] = np.nan
              if (row.is_inhibited== ""):
                df.loc[row.Index,'flag'] = ' check is_inhibited;'
              if (row.is_inhibited== "unknown"):
                df.loc[row.Index,'flag'] = ' test for inhibition has not been performed;'

   # days prior to conc_extract
   e="transit_days"
   if e in dic_name:
       for row in df.itertuples():
           if (row.Target!= 'Xeno')&(row.date_conc_extract !="")&(row.date_sampling !=""):
                 if (((row.stored_minus_80 != '0')& (row.stored_minus_80 != ''))| ((row.stored_minus_20 != '0')& (row.stored_minus_20 != ''))):
                   df.loc[row.Index,'quality_score'] = np.nan
                   df.loc[row.Index,'flag'] = ' Sample was frozen prior to concentration and extraction;'
                 else:
                   days_transit=pd.to_datetime(row.date_conc_extract)-pd.to_datetime(row.date_sampling)
                   if days_transit/np.timedelta64(1, 'D') < 0:
                     df.loc[row.Index,'quality_score'] = np.nan
                     df.loc[row.Index,'flag'] = " date_conc_extract < date_sampling;"
                   else:
                      if (days_transit/np.timedelta64(1, 'D') >=3):
                        value= row.quality_score + p[e][0]*p[e][1]
                        df.loc[row.Index,'quality_score'] = value
                      elif (days_transit/np.timedelta64(1, 'D') >=5):
                        value= row.quality_score + p[e][0]*p[e][2]
                        df.loc[row.Index,'quality_score'] = value
                        df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " days before concentration/extraction (2);"
                      else:
                        value= row.quality_score  + p[e][0]*p[e][3]
                        df.loc[row.Index,'quality_score'] = value
                        df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " days before concentration/extraction (3);"
                        if row.replicate_count==0:
                          df.loc[row.Index,'flag'] = 'set to 0'
                          df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " 0 replicates;"
           else:
              df.loc[row.Index,'quality_score'] = np.nan
              if np.isnan(row.replicate_count):
                df.loc[row.Index,'flag'] = df.loc[row.Index,'flag'] + " check date_conc_extract and date_sampling;"

   #PBS control
   e="PBS_amp"
   if e in dic_name:
        for row in df.itertuples():
           if (row.Target!= 'Xeno')&(row.PBS_result!= ""):
                  if (row.PBS_result =='negative'):
                    value= row.quality_score + p[e][0]*p[e][1]
                    df.loc[row.Index,'quality_score'] = value
                  elif (pd.to_numeric(row.PBS_result) < pd.to_numeric(row.Cq_of_lowest_std_quantity)+1):
                    value= row.quality_score + p[e][0]*p[e][2]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " PBS (2);"
                  else:
                    value= row.quality_score  + p[e][0]*p[e][3]
                    df.loc[row.Index,'quality_score'] = value
                    df.loc[row.Index,'point_deduction'] = df.loc[row.Index,'point_deduction'] + " PBS (3);"
           else:
              df.loc[row.Index,'quality_score'] = np.nan
              if (row.ntc_result== ""):
                df.loc[row.Index,'flag'] = df.loc[row.Index,'flag'] + " check PBS batch is correctly assigned;"

   for row in df.itertuples():
       if row.flag=='set to 0':
           df.loc[row.Index,'quality_score'] = 0

   df.quality_score=(df.quality_score/max_score)*100
   return df
