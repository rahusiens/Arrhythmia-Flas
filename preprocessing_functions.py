import neurokit2 as nk
import pandas as pd
import numpy as np
import pywt
from datetime import datetime, time

window_length = 5
min_size = 0

def calc_baseline(signal):
    """
    Calculate the baseline of signal.
    Args:
        signal (numpy 1d array): signal whose baseline should be calculated
    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]

def fix_time(time_str):
  list_string = str(time_str).split(".", 1)
  if (len(list_string[1]) > 3):
    return time_str[:-1]
  else:
    return time_str

def reshape_indexes(*arguments):
  global min_size
  min_size = min(map(len, arguments))

def calc_mean(lst,window_length):
  list_sum = np.add.reduceat(lst, np.arange(0, len(lst), window_length))
  remainder = len(list_sum) % window_length
  list_mean = []
  for i in range(len(list_sum)):
    if (i == len(list_sum)-1 and remainder > 0):
      list_mean.append(list_sum[i] / remainder)
    else:
      list_mean.append(list_sum[i] / window_length)
  return list_mean

def calc_r_peaks(ecg_cleaned, sample_rate):
  cg_signals, info = nk.ecg_peaks(ecg_cleaned, method='pantompkins', sampling_rate=sample_rate)
  rpeaks = [x for x in info["ECG_R_Peaks"] if np.isnan(x) == False]
  return rpeaks

def calc_pqst_peaks(ecg_cleaned, rpeaks, sample_rate):
  _, pqst_peaks = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=sample_rate, method="peak")
  return pqst_peaks

def calc_pqrs_stats(ecg_cleaned,pqst_peaks, rpeaks):
  p_index = [x for x in pqst_peaks["ECG_P_Peaks"] if np.isnan(x) == False]
  q_index = [x for x in pqst_peaks["ECG_Q_Peaks"] if np.isnan(x) == False]
  s_index = [x for x in pqst_peaks["ECG_S_Peaks"] if np.isnan(x) == False]

  p_data = ecg_cleaned[p_index].values.tolist()
  q_data = ecg_cleaned[q_index].values.tolist()
  s_data = ecg_cleaned[s_index].values.tolist()
  r_data = ecg_cleaned[rpeaks].values.tolist()

  p_data_mean = calc_mean(p_data,window_length)
  q_data_mean = calc_mean(q_data,window_length)
  r_data_mean = calc_mean(r_data,window_length)
  s_data_mean = calc_mean(s_data,window_length)
  r_max = np.maximum.reduceat(r_data, np.arange(0, len(r_data), window_length))
  r_min = np.minimum.reduceat(r_data, np.arange(0, len(r_data), window_length))

  return p_data_mean,q_data_mean,r_data_mean,s_data_mean,r_max,r_min
  # return abs(np.mean(p_data)), abs(np.mean(q_data)), abs(np.mean(r_data)), abs(r_max), abs(r_min), abs(np.mean(s_data))

def calc_rr_interval(ecg_time, rpeaks):
  #clean time column string
  chars_to_remove = [' ', '[', ']', "'"]
  sc = set(chars_to_remove)

  r_peaks = ecg_time[rpeaks].values.tolist()

  rr_interval = []

  for i in range(len(r_peaks)-1):
    r_time_1 = r_peaks[i]
    r_time_2 = r_peaks[i+1]
    r_time_1_str = ''.join([c for c in r_time_1 if c not in sc])
    r_time_2_str = ''.join([c for c in r_time_2 if c not in sc])
    r_time_1_str = fix_time(r_time_1_str)
    r_time_2_str = fix_time(r_time_2_str)
    try:
      r_time_1_object = datetime.strptime(r_time_1_str, '%H:%M:%S.%f')
    except:
      r_time_1_object = datetime.strptime(r_time_1_str, '%M:%S.%f')

    try:
      r_time_2_object = datetime.strptime(r_time_2_str, '%H:%M:%S.%f')
    except:
      r_time_2_object = datetime.strptime(r_time_2_str, '%M:%S.%f')

    #r-peak 2 time - r-peak 1 time
    duration = r_time_2_object - r_time_1_object
    #save result to list
    rr_interval.append(abs(duration.total_seconds()))

  rr_interval_mean = calc_mean(rr_interval,window_length)
  return rr_interval_mean
  # return abs(np.mean(rr_interval))

def get_signal_time_data(signal, time, rpeaks):
  signal_time = time[rpeaks].values.tolist()
  print(time[0])
  chars_to_remove = [' ', '[', ']', "'"]
  sc = set(chars_to_remove)

  for i in range(len(signal_time)):
    r_time_1 = signal_time[i]
    r_time_1_str = ''.join([c for c in r_time_1 if c not in sc])
    r_time_1_str = fix_time(r_time_1_str)
    print(r_time_1_str)
    signal_time[i] = r_time_1_str

  time_list = []
  signal_list = []
  print(time[0])

  if len(signal_time) > 5:
    for i in range(0, len(signal_time), window_length):
      start = signal_time[i]
      remainder = len(signal_time) % window_length
      try:
        end = signal_time[i+5]
      except:
        end = signal_time[i+remainder-1]

      try:
        start_object = datetime.strptime(start, '%H:%M:%S.%f')
      except:
        start_object = datetime.strptime(start, '%M:%S.%f')
      try:
        end_object = datetime.strptime(end, '%H:%M:%S.%f')
      except:
        end_object = datetime.strptime(end, '%M:%S.%f')


      if start_object > end_object:
        temp = end
        end = start
        start = temp

      time_data = time[time.between(start, end)].values.tolist()
      signal_data = signal[time.between(start, end)].values.tolist()
      time_list.append(time_data)
      signal_list.append(signal_data)

  # print(time_list, signal_list)
  return time_list, signal_list

def calc_qrs_complex(ecg_time, pqst_peaks):
  chars_to_remove = [' ', '[', ']', "'"]
  sc = set(chars_to_remove)
  q_index = [x for x in pqst_peaks["ECG_Q_Peaks"] if np.isnan(x) == False]
  s_index = [x for x in pqst_peaks["ECG_S_Peaks"] if np.isnan(x) == False]
  q_peaks = ecg_time.loc[q_index].values.tolist()
  s_peaks = ecg_time.loc[s_index].values.tolist()

  length = min(len(q_peaks), len(s_peaks))
  qrs_complex = []
  for i in range(length):
    q_time = q_peaks[i]
    s_time = s_peaks[i]
    #clean time column string
    q_time_str = ''.join([c for c in q_time if c not in sc])
    s_time_str = ''.join([c for c in s_time if c not in sc])
    q_time_str = fix_time(q_time_str)
    s_time_str = fix_time(s_time_str)
    try:
      q_time_object = datetime.strptime(q_time_str, '%H:%M:%S.%f')
    except:
      q_time_object = datetime.strptime(q_time_str, '%M:%S.%f')
    try:
      s_time_object = datetime.strptime(s_time_str, '%H:%M:%S.%f')
    except:
      s_time_object = datetime.strptime(s_time_str, '%M:%S.%f')
    #s-peak time - q-peak time
    duration = s_time_object - q_time_object
    #save result to list
    qrs_complex.append(abs(duration.total_seconds()))
  if(len(qrs_complex) > 1):
    qrs_complex.pop(0)

  qrs_complex_mean = calc_mean(qrs_complex,window_length)
  return qrs_complex_mean
  # return abs(np.mean(qrs_complex))