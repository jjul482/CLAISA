
"""
A script to merge AIS messages into AIS tracks.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#sys.path.append("..")
#import utils
import pickle
import copy
import csv
from datetime import datetime
import time
from io import StringIO
from tqdm import tqdm as tqdm
from dateutil import parser

D2C_MIN = 2000 #meters

#===============
# ## Gulf of Mexico full
# LAT_MIN = 17.4068
# LAT_MAX = 31.4648
# LON_MIN = -98.0539
# LON_MAX = -80.4332

#===============
# ## Gulf of Mexico
# LAT_MIN = 27.0
# LAT_MAX = 30.0
# LON_MIN = -90.5
# LON_MAX = -87.5

#===============
# ## East Coast full
# LAT_MIN = 26.23
# LAT_MAX = 45.67
# LON_MIN = -81.51
# LON_MAX = -50.99

#===============
# ## East Coast
LAT_MIN = 29.4
LAT_MAX = 32.4
LON_MIN = -81.3
LON_MAX = -78.3

dataset_name = 'eastcoast'
dataset_path = "data/"
l_csv_filename =[] # USA datasets

#origin_directory = 'C://AIS_Datasets//toprocess'
origin_directory = 'data'
for root, dirs, files in os.walk(origin_directory):
    for file in files:
        if file.endswith('.csv'):
            print(f"Detected file {file}")
            l_csv_filename.append(file)

#l_csv_filename =["Est-aruba_5x5deg_2018001_2018180.csv"]
pkl_filename = dataset_name + "_2020_03_20_COVID.pkl"
pkl_filename_train = dataset_name + "_01_2018_train.pkl"
pkl_filename_valid = dataset_name + "_01_2018_valid.pkl"
pkl_filename_test  = dataset_name + "_2020_03_20_COVID_test.pkl"

cargo_tanker_filename = "AIS_2020_03_20_COVID_cargo_tanker.npy"

t_train_min = time.mktime(time.strptime("01/01/2018 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_train_max = time.mktime(time.strptime("26/01/2018 17:59:59", "%d/%m/%Y %H:%M:%S"))
t_valid_min = time.mktime(time.strptime("26/01/2018 18:00:00", "%d/%m/%Y %H:%M:%S"))
t_valid_max = time.mktime(time.strptime("01/02/2018 19:59:59", "%d/%m/%Y %H:%M:%S"))
t_test_min  = time.mktime(time.strptime("20/03/2020 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_test_max  = time.mktime(time.strptime("20/03/2020 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("31/01/2021 23:59:59", "%d/%m/%Y %H:%M:%S"))

#========================================================================
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SOG_MAX = 30.0  # the SOG is truncated to 30.0 knots max.

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, TIMESTAMP, MMSI, SHIPTYPE, STATUS  = list(range(8))

CARGO_TANKER_ONLY = True
print(pkl_filename_train)


## LOADING CSV FILES
#======================================
l_l_msg = [] # list of AIS messages, each row is a message (list of AIS attributes)
n_error = 0
for csv_filename in l_csv_filename:
    data_path = os.path.join(origin_directory,csv_filename)
    with open(data_path,"r") as f:
        print("Reading ", csv_filename, "...")
        csvReader = csv.reader(f)
        next(csvReader) # skip the legend row
        count = 1
        for row in tqdm(csvReader):
#             utc_time = datetime.strptime(row[8], "%Y/%m/%d %H:%M:%S")
#             timestamp = (utc_time - EPOCH).total_seconds()
            # Format: LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, STATUS
            count += 1
            try:
                l_l_msg.append([float(row[2]),float(row[3]),
                                float(row[4]),float(row[5]),
                                time.mktime(parser.parse(row[1]).timetuple()),
                                int(row[0]),int(row[10]),int(row[11])])
            except:
                n_error += 1
                continue


m_msg = np.array(l_l_msg)
#del l_l_msg
print("Total number of AIS messages: ",m_msg.shape[0])

print("Lat min: ",np.min(m_msg[:,LAT]), "Lat max: ",np.max(m_msg[:,LAT]))
print("Lon min: ",np.min(m_msg[:,LON]), "Lon max: ",np.max(m_msg[:,LON]))
print("Ts min: ",np.min(m_msg[:,TIMESTAMP]), "Ts max: ",np.max(m_msg[:,TIMESTAMP]))

if m_msg[0,TIMESTAMP] > 1584720228: 
    m_msg[:,TIMESTAMP] = m_msg[:,TIMESTAMP]/1000 # Convert to suitable timestamp format

print("Time min: ",datetime.utcfromtimestamp(np.min(m_msg[:,TIMESTAMP])).strftime('%Y-%m-%d %H:%M:%SZ'))
print("Time max: ",datetime.utcfromtimestamp(np.max(m_msg[:,TIMESTAMP])).strftime('%Y-%m-%d %H:%M:%SZ'))
if CARGO_TANKER_ONLY:
    m_msg = m_msg[m_msg[:, SHIPTYPE] >= 70]
    m_msg = m_msg[m_msg[:, SHIPTYPE] <= 90]
    np.save("cargo_temp.npy", m_msg)

## Vessel Type    
#======================================
# print("Selecting vessel type ...")
# def sublist(lst1, lst2):
#    ls1 = [element for element in lst1 if element in lst2]
#    ls2 = [element for element in lst2 if element in lst1]
#    return (len(ls1) != 0) and (ls1 == ls2)

# VesselTypes = dict()
# l_mmsi = []
# n_error = 0
# for v_msg in tqdm(m_msg):
#     try:
#         mmsi_ = v_msg[MMSI]
#         type_ = v_msg[SHIPTYPE]
#         if mmsi_ not in l_mmsi :
#             VesselTypes[mmsi_] = [type_]
#             l_mmsi.append(mmsi_)
#         elif type_ not in VesselTypes[mmsi_]:
#             VesselTypes[mmsi_].append(type_)
#     except:
#         n_error += 1
#         continue
# print(n_error)
# for mmsi_ in tqdm(list(VesselTypes.keys())):
#     VesselTypes[mmsi_] = np.sort(VesselTypes[mmsi_])
    
# l_cargo_tanker = []
# l_fishing = []
# for mmsi_ in list(VesselTypes.keys()):
#     if sublist(VesselTypes[mmsi_], list(range(70,80))) or sublist(VesselTypes[mmsi_], list(range(80,90))):
#         l_cargo_tanker.append(mmsi_)
#     if sublist(VesselTypes[mmsi_], [30]):
#         l_fishing.append(mmsi_)

# print("Total number of vessels: ",len(VesselTypes))
# print("Total number of cargos/tankers: ",len(l_cargo_tanker))
# print("Total number of fishing: ",len(l_fishing))

# print("Saving vessels' type list to ", cargo_tanker_filename)
# np.save(cargo_tanker_filename,l_cargo_tanker)
# np.save(cargo_tanker_filename.replace("_cargo_tanker.npy","_fishing.npy"),l_fishing)


## FILTERING 
#======================================
# Selecting AIS messages in the ROI and in the period of interest.

## LAT LON
m_msg = m_msg[m_msg[:,LAT]>=LAT_MIN]
m_msg = m_msg[m_msg[:,LAT]<=LAT_MAX]
m_msg = m_msg[m_msg[:,LON]>=LON_MIN]
m_msg = m_msg[m_msg[:,LON]<=LON_MAX]
# SOG
m_msg = m_msg[m_msg[:,SOG]>=0]
m_msg = m_msg[m_msg[:,SOG]<=SOG_MAX]
# COG
m_msg = m_msg[m_msg[:,SOG]>=0]
m_msg = m_msg[m_msg[:,COG]<=360]

# TIME
m_msg = m_msg[m_msg[:,TIMESTAMP]>=0]

m_msg = m_msg[m_msg[:,TIMESTAMP]>=t_min]
m_msg = m_msg[m_msg[:,TIMESTAMP]<=t_max]
m_msg_train = m_msg[m_msg[:,TIMESTAMP]>=t_train_min]
m_msg_train = m_msg_train[m_msg_train[:,TIMESTAMP]<=t_train_max]
m_msg_valid = m_msg[m_msg[:,TIMESTAMP]>=t_valid_min]
m_msg_valid = m_msg_valid[m_msg_valid[:,TIMESTAMP]<=t_valid_max]
m_msg_test  = m_msg[m_msg[:,TIMESTAMP]>=t_test_min]
m_msg_test  = m_msg_test[m_msg_test[:,TIMESTAMP]<=t_test_max]

print("Total msgs: ",len(m_msg))
print("Number of msgs in the training set: ",len(m_msg_train))
print("Number of msgs in the validation set: ",len(m_msg_valid))
print("Number of msgs in the test set: ",len(m_msg_test))


## MERGING INTO DICT
#======================================
# Creating AIS tracks from the list of AIS messages.
# Each AIS track is formatted by a dictionary.
print("Convert to dicts of vessel's tracks...")

# Training set
Vs_train = dict()
for v_msg in tqdm(m_msg_train):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_train.keys())):
        Vs_train[mmsi] = np.empty((0,8))
    Vs_train[mmsi] = np.concatenate((Vs_train[mmsi], np.expand_dims(v_msg[:9],0)), axis = 0)
for key in tqdm(list(Vs_train.keys())):
    Vs_train[key] = np.array(sorted(Vs_train[key], key=lambda m_entry: m_entry[TIMESTAMP]))

# Validation set
Vs_valid = dict()
for v_msg in tqdm(m_msg_valid):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_valid.keys())):
        Vs_valid[mmsi] = np.empty((0,8))
    Vs_valid[mmsi] = np.concatenate((Vs_valid[mmsi], np.expand_dims(v_msg[:9],0)), axis = 0)
for key in tqdm(list(Vs_valid.keys())):
    Vs_valid[key] = np.array(sorted(Vs_valid[key], key=lambda m_entry: m_entry[TIMESTAMP]))

# Test set
Vs_test = dict()
for v_msg in tqdm(m_msg_test):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_test.keys())):
        Vs_test[mmsi] = np.empty((0,8))
    Vs_test[mmsi] = np.concatenate((Vs_test[mmsi], np.expand_dims(v_msg[:9],0)), axis = 0)
for key in tqdm(list(Vs_test.keys())):
    Vs_test[key] = np.array(sorted(Vs_test[key], key=lambda m_entry: m_entry[TIMESTAMP]))


## PICKLING
#======================================
for filename, filedict in zip([pkl_filename_train,pkl_filename_valid,pkl_filename_test], 
                              [Vs_train,Vs_valid,Vs_test]
                             ):
    print("Writing to ", os.path.join(dataset_path,filename),"...")
    with open(os.path.join(dataset_path,filename),"wb") as f:
        pickle.dump(filedict,f)
    print("Total number of tracks: ", len(filedict))
