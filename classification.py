import neurokit2 as nk
import pandas as pd
import joblib
import pickle
from preprocessing_functions import *

window_length = 5
min_size = 0
features = ['rr_interval',	'qrs_complex']
scaler = joblib.load("scaler_2.bin")

def cek(ecg):
  feature = preprocess(ecg)
  feature_normalized = scaler.transform(feature[features])
  return feature_normalized


def preprocess(ecg):
  features1_total = pd.DataFrame()
  features2_total = pd.DataFrame()
  features_2 = pd.DataFrame()

  data = ecg

  time = data["time"]
  signal1 = data["lead1"]
  signal2 = data["lead2"]
  # annotation = data["Aux"][0]
  # annotation = re.sub("[0\t(]", "", annotation)

  baseline1 = calc_baseline(signal1)
  baseline2 = calc_baseline(signal2)
  ecg_out1 = signal1 - baseline1
  ecg_out2 = signal2 - baseline2

  #get r-peaks and rr-interval
  rpeaks1 = calc_r_peaks(ecg_out1, 128)
  rr_interval1 = calc_rr_interval(time, rpeaks1)

  rpeaks2 = calc_r_peaks(ecg_out2, 128)
  rr_interval2 = calc_rr_interval(time,rpeaks2)

  #get p, q, s, and t peaks
  pqst_peaks1 = calc_pqst_peaks(ecg_out1, rpeaks1, 128)
  pqst_peaks2 = calc_pqst_peaks(ecg_out2, rpeaks2, 128)

  #calculate qrs complex duration
  qrs_complex1 = calc_qrs_complex(time, pqst_peaks1)
  qrs_complex2 = calc_qrs_complex(time, pqst_peaks2)\

  #calculate pqrs stats
  p_means1, q_means1, r_means1, r_max1, r_min1, s_means1 = calc_pqrs_stats(ecg_out1,pqst_peaks1, rpeaks1)
  p_means2, q_means2, r_means2, r_max2, r_min2, s_means2 = calc_pqrs_stats(ecg_out2,pqst_peaks2, rpeaks2)

  # #numeric annotation
  # if annotation != None and "AF" in annotation:
  #   target = 2
  # elif annotation != None and "N" in annotation:
  #   target = 1
  # else:
  #   target = 0

  # annotation_list1 = [annotation] * len(rr_interval1)
  # target_list1 = [target] * len(rr_interval1)
  # annotation_list2 = [annotation] * len(rr_interval2)
  # target_list2 = [target] * len(rr_interval2)

  #save rr-interval and qrs complex to a folder
  details_1 = {
  'rr_interval' : rr_interval1,
  'qrs_complex' : qrs_complex1,
  'p_means' : p_means1,
  'q_means' : q_means1,
  'r_means' : r_means1,
  'r_max' : r_max1,
  'r_min' : r_min1,
  's_means' : s_means1,
  # 'annotation' : annotation_list1,
  # 'target' : target_list1,
  # 'data_source' : folder
  }

  details_2 = {
  'rr_interval' : rr_interval2,
  'qrs_complex' : qrs_complex2,
  # 'p_means' : p_means2,
  # 'q_means' : q_means2,
  # 'r_means' : r_means2,
  # 'r_max' : r_max2,
  # 'r_min' : r_min2,
  # 's_means' : s_means2,
  # 'annotation' : annotation_list2,
  # 'target' : target_list2,
  # 'data_source' : folder
  }

  # features = pd.concat([df_rr_interval_1,df_rr_interval_2,df_qrs_complex_1,df_qrs_complex_2,df_target], ignore_index=True, axis=1)
  features_1 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in details_1.items() ]))
  features_2 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in details_2.items() ]))
  # features_1['annotation'].fillna(method='pad', inplace=True)
  # features_2['annotation'].fillna(method='pad', inplace=True)
  # features_1['target'].fillna(method='pad', inplace=True)
  # features_2['target'].fillna(method='pad', inplace=True)
  # features_1['data_source'].fillna(method='pad', inplace=True)
  # features_2['data_source'].fillna(method='pad', inplace=True)


  # reshape_indexes(features_1,features_2)
  # features_1 = features_1[:min_size]
  # features_2 = features_2[:min_size]

  
  # signal_time1 = time[rpeaks1].values.tolist()
  # signal_time2 = time[rpeaks2].values.tolist()
  # signal_time1 = [signal_time1[k:k + window_length] for k in range(0, len(signal_time1), window_length)]
  # signal_time2 = [signal_time2[k:k + window_length] for k in range(0, len(signal_time2), window_length)]

  # features_1.loc[:,'time'] = pd.Series(signal_time1)
  # features_2.loc[:,'time'] = pd.Series(signal_time2)

  # features1_total = features1_total.append(features_1, ignore_index = True)
  # features2_total = features2_total.append(features_2, ignore_index = True)
  return features_2