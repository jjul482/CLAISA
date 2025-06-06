
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import sys
import os
from tqdm import tqdm_notebook as tqdm
import utils
import pickle
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import time
from io import StringIO

from tqdm import tqdm
import argparse

# In[2]:
def getConfig(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    # Gulf of Mexico full
    # parser.add_argument("--lat_min", type=float, default=17.4068,
    #                     help="Lat min.")
    # parser.add_argument("--lat_max", type=float, default=31.4648,
    #                     help="Lat max.")
    # parser.add_argument("--lon_min", type=float, default=-98.0539,
    #                     help="Lon min.")
    # parser.add_argument("--lon_max", type=float, default=-80.4332,
    #                     help="Lon max.")

    # Gulf of Mexico
    # parser.add_argument("--lat_min", type=float, default=27.0,
    #                     help="Lat min.")
    # parser.add_argument("--lat_max", type=float, default=30.0,
    #                     help="Lat max.")
    # parser.add_argument("--lon_min", type=float, default=-90.5,
    #                     help="Lon min.")
    # parser.add_argument("--lon_max", type=float, default=-87.5,
    #                     help="Lon max.")

    # East Coast full
    # parser.add_argument("--lat_min", type=float, default=26.23,
    #                     help="Lat min.")
    # parser.add_argument("--lat_max", type=float, default=45.67,
    #                     help="Lat max.")
    # parser.add_argument("--lon_min", type=float, default=-81.51,
    #                     help="Lon min.")
    # parser.add_argument("--lon_max", type=float, default=-50.99,
    #                     help="Lon max.")

    # East Coast
    # parser.add_argument("--lat_min", type=float, default=29.4,
    #                     help="Lat min.")
    # parser.add_argument("--lat_max", type=float, default=32.4,
    #                     help="Lat max.")
    # parser.add_argument("--lon_min", type=float, default=-81.3,
    #                     help="Lon min.")
    # parser.add_argument("--lon_max", type=float, default=-78.3,
    #                     help="Lon max.")
    # Piraeus
    # parser.add_argument("--lat_min", type=float, default=37.5,
    #                     help="Lat min.")
    # parser.add_argument("--lat_max", type=float, default=38.1,
    #                     help="Lat max.")
    # parser.add_argument("--lon_min", type=float, default=23,
    #                     help="Lon min.")
    # parser.add_argument("--lon_max", type=float, default=23.9,
    #                     help="Lon max.")
    # Denmark
    parser.add_argument("--lat_min", type=float, default=55.0,
                        help="Lat min.")
    parser.add_argument("--lat_max", type=float, default=58.0,
                        help="Lat max.")
    parser.add_argument("--lon_min", type=float, default=10.0,
                        help="Lon min.")
    parser.add_argument("--lon_max", type=float, default=13.0,
                        help="Lon max.")
    
    dataset_name = "denmark"
    # File paths
    parser.add_argument("--dataset_dir", type=str, 
                        default=f"data/{dataset_name}",
                        help="Dir to dataset.")    
    parser.add_argument("--l_input_filepath", type=str, nargs='+',
                        #default=["eastcoast_01_2018_test.pkl"],
                        default=[f"{dataset_name}_2019_test.pkl"],
                        help="List of path to input files.")
    parser.add_argument("--output_filepath", type=str,
                        default=f"data/ct_dma/{dataset_name}/ct_dma_test.pkl",
                        help="Path to output file.")
    
    parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    config = parser.parse_args(args)
    return config

config = getConfig(sys.argv[1:])

#=====================================================================
LAT_MIN,LAT_MAX,LON_MIN,LON_MAX = config.lat_min,config.lat_max,config.lon_min,config.lon_max

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots
DURATION_MAX = 24 #h
min_len = 0.5 #h

EPOCH = datetime(1970, 1, 1)
features = 7
LAT, LON, SOG, COG, TIMESTAMP, MMSI, STATUS  = list(range(features))

FIG_W = 960
FIG_H = int(960*LAT_RANGE/LON_RANGE) #533 #768

dict_list = []
for filename in config.l_input_filepath:
    with open(os.path.join(config.dataset_dir,filename),"rb") as f:
        temp = pickle.load(f)
        dict_list.append(temp)
# In[3]:

print(" Remove erroneous timestamps and erroneous speeds...")
Vs = dict()
for Vi,filename in zip(dict_list, config.l_input_filepath):
    print(filename)
    for mmsi in list(Vi.keys()):       
        # Boundary
        lat_idx = np.logical_or((Vi[mmsi][:,LAT] > LAT_MAX),
                                (Vi[mmsi][:,LAT] < LAT_MIN))
        Vi[mmsi] = Vi[mmsi][np.logical_not(lat_idx)]
        lon_idx = np.logical_or((Vi[mmsi][:,LON] > LON_MAX),
                                (Vi[mmsi][:,LON] < LON_MIN))
        Vi[mmsi] = Vi[mmsi][np.logical_not(lon_idx)]
#         # Abnormal timestamps
#         abnormal_timestamp_idx = np.logical_or((Vi[mmsi][:,TIMESTAMP] > t_max),
#                                                (Vi[mmsi][:,TIMESTAMP] < t_min))
#         Vi[mmsi] = Vi[mmsi][np.logical_not(abnormal_timestamp_idx)]
        # Abnormal speeds
        abnormal_speed_idx = Vi[mmsi][:,SOG] > SPEED_MAX
        Vi[mmsi] = np.delete(Vi[mmsi][np.logical_not(abnormal_speed_idx)], 6, 1)
        # Deleting empty keys
        if len(Vi[mmsi]) == 0:
            del Vi[mmsi]
            continue
        if mmsi not in list(Vs.keys()):
            Vs[mmsi] = Vi[mmsi]
            del Vi[mmsi]
        else:
            Vs[mmsi] = np.concatenate((Vs[mmsi],Vi[mmsi]),axis = 0)
            del Vi[mmsi]
del dict_list, Vi, abnormal_speed_idx

## STEP 2: VOYAGES SPLITTING 
#======================================
# Cutting discontiguous voyages into contiguous ones
print("Cutting discontiguous voyages into contiguous ones...")
count = 0
voyages = dict()
INTERVAL_MAX = 2*3600 # 2h
for mmsi in list(Vs.keys()):
    v = Vs[mmsi]
    # Intervals between successive messages in a track
    intervals = v[1:,TIMESTAMP] - v[:-1,TIMESTAMP]
    idx = np.where(intervals > INTERVAL_MAX)[0]
    if len(idx) == 0:
        voyages[count] = v
        count += 1
    else:
        tmp = np.split(v,idx+1)
        for t in tmp:
            voyages[count] = t
            count += 1

print(len(voyages))


# In[7]:


# STEP 3: REMOVING SHORT VOYAGES
#======================================
# Removing AIS track whose length is smaller than 20 or those last less than 4h
print("Removing AIS track whose length is smaller than 20 or those last less than 4h...")

for k in list(voyages.keys()):
    duration = voyages[k][-1,TIMESTAMP] - voyages[k][0,TIMESTAMP]
    if (len(voyages[k]) < 20) or (duration < min_len*(3600*1)):
        voyages.pop(k, None)


print(len(voyages))
# In[9]:


# STEP 4: REMOVING OUTLIERS
#======================================
# print("Removing anomalous message...")
# error_count = 0
# tick = time.time()
# for k in  tqdm(list(voyages.keys())):
#     track = voyages[k][:,[TIMESTAMP,LAT,LON,SOG]] # [Timestamp, Lat, Lon, Speed]
#     try:
#         o_report, o_calcul = utils.detectOutlier(track, speed_max = 30)
#         if o_report.all() or o_calcul.all():
#             voyages.pop(k, None)
#         else:
#             voyages[k] = voyages[k][np.invert(o_report)]
#             voyages[k] = voyages[k][np.invert(o_calcul)]
#     except:
#         voyages.pop(k,None)
#         error_count += 1
# tok = time.time()
print("Removing anomalous messages...")
error_count = 0
tick = time.time()

voyages_cleaned = {}
for k in tqdm(voyages.keys()):
    track = voyages[k][:, [TIMESTAMP, LAT, LON, SOG]]
    try:
        o_report, o_calcul = utils.detectOutlier_fast(track, speed_max=30)
        combined_mask = ~(o_report | o_calcul)
        if not combined_mask.any():
            continue  # skip this voyage
        voyages_cleaned[k] = voyages[k][combined_mask]
    except Exception:
        error_count += 1
        continue

voyages = voyages_cleaned
tok = time.time()

print(len(voyages))
print("STEP 4: duration = ",(tok - tick)/60) # 139.685766101


# In[13]:


## STEP 6: SAMPLING
#======================================
# Sampling, resolution = 1 min
# print('Sampling...')
# Vs = dict()
# count = 0
# for k in tqdm(list(voyages.keys())):
#     v = voyages[k]
#     sampling_track = np.empty((0, features))
#     for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), 60): # 1 min
#         tmp = utils.interpolate(t,v)
#         if tmp is not None:
#             sampling_track = np.vstack([sampling_track, tmp])
#         else:
#             sampling_track = None
#             break
#     if sampling_track is not None:
#         Vs[count] = sampling_track
#         count += 1
print("Sampling...")
Vs = {}
count = 0

for k in tqdm(voyages):
    v = voyages[k]
    start_time = int(v[0, TIMESTAMP])
    end_time = int(v[-1, TIMESTAMP])
    
    sampling_track = []
    failed = False

    for t in range(start_time, end_time, 60):  # 1 min
        tmp = utils.interpolate(t, v)
        if tmp is None:
            failed = True
            break
        sampling_track.append(tmp)

    if not failed:
        Vs[count] = np.array(sampling_track)
        count += 1

## STEP 8: RE-SPLITTING
#======================================
print('Re-Splitting...')
Data = dict()
count = 0
for k in tqdm(list(Vs.keys())): 
    v = Vs[k]
    # Split AIS track into small tracks whose duration <= 1 day
    idx = np.arange(0, len(v), 12*DURATION_MAX)[1:]
    tmp = np.split(v,idx)
    for subtrack in tmp:
        # only use tracks whose duration >= 4 hours
        if len(subtrack) >= 12*4:
            Data[count] = subtrack
            count += 1
print(len(Data))


# ## STEP 5: REMOVING 'MOORED' OR 'AT ANCHOR' VOYAGES
# #======================================
# # Removing 'moored' or 'at anchor' voyages
# print("Removing 'moored' or 'at anchor' voyages...")
# for mmsi in  tqdm(list(voyages.keys())):
#     d_L = float(len(voyages[mmsi]))

#     if np.count_nonzero(voyages[mmsi][:,NAV_STT] == 1)/d_L > 0.7       or np.count_nonzero(voyages[mmsi][:,NAV_STT] == 5)/d_L > 0.7:
#         voyages.pop(mmsi,None)
#         continue
#     sog_max = np.max(voyages[mmsi][:,SOG])
#     if sog_max < 1.0:
#         voyages.pop(mmsi,None)

        
## STEP 5: REMOVING 'MOORED' OR 'AT ANCHOR' VOYAGES
#======================================
# Removing 'moored' or 'at anchor' voyages
print("Removing 'moored' or 'at anchor' voyages...")
for k in  tqdm(list(Data.keys())):
    d_L = float(len(Data[k]))

    if np.count_nonzero(Data[k][:,STATUS] == 1)/d_L > 0.7 \
    or np.count_nonzero(Data[k][:,STATUS] == 5)/d_L > 0.7:
        Data.pop(k,None)
        continue
    sog_max = np.max(Data[k][:,SOG])
    if sog_max < 1.0:
        Data.pop(k,None)
print(len(Data))
# In[12]:


# In[15]:


## STEP 6: REMOVING LOW SPEED TRACKS
#======================================
print("Removing 'low speed' tracks...")
for k in tqdm(list(Data.keys())):
    d_L = float(len(Data[k]))
    if np.count_nonzero(Data[k][:,SOG] < 2)/d_L > 0.8:
        Data.pop(k,None)
print(len(Data))


# In[21]:


## STEP 9: NORMALISATION
#======================================
print('Normalisation...')
for k in tqdm(list(Data.keys())):
    v = Data[k]
    v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
    v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
    v[:,SOG][v[:,SOG] > SPEED_MAX] = SPEED_MAX
    v[:,SOG] = v[:,SOG]/SPEED_MAX
    v[:,COG] = v[:,COG]/360.0


# In[22]:


print(config.output_filepath)


# In[23]:


# plt.plot(Data[0][:,LON],Data[0][:,LAT])


# In[24]:


print(len(Data))


# In[25]:


print(os.path.dirname(config.output_filepath))


# In[26]:


os.path.exists(os.path.dirname(config.output_filepath))


# In[27]:


if not os.path.exists(os.path.dirname(config.output_filepath)):
    os.makedirs(os.path.dirname(config.output_filepath))


# In[28]:
traj_dict = []
for key in Data.keys():
    traj_dict.append({'mmsi' : key, 'traj' : np.array(Data[key])})


## STEP 10: WRITING TO DISK
#======================================
with open(config.output_filepath,"wb") as f:
    pickle.dump(traj_dict,f)


# In[29]:


# print(debug)


# In[30]:


print(len(Data))


# In[31]:


minlen = 1000
for k in list(Data.keys()):
    v = Data[k]
    if len(v) < minlen:
        minlen = len(v)
print("min len: ",minlen)


# In[32]:


# len(Data[0])


# In[33]:


# print(debug)


# In[34]:


## Loading coastline polygon.
# For visualisation purpose, delete this part if you do not have coastline
# shapfile

coastline_filename = "./streetmap_coastline_Bretagne.pkl"

if "bretagne" in config.output_filepath:
    with open(coastline_filename, 'rb') as f:
        l_coastline_poly = pickle.load(f)

# In[35]:


config.output_filepath


# In[36]:


Vs = Data
FIG_DPI = 150
plt.figure(figsize=(FIG_W/FIG_DPI, FIG_H/FIG_DPI), dpi=FIG_DPI)
cmap = plt.cm.get_cmap('Blues')
l_keys = list(Vs.keys())
N = len(Vs)
for d_i in range(N):
    key = l_keys[d_i]
    c = cmap(float(d_i)/(N-1))
    tmp = Vs[key]
    v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
    v_lon = tmp[:,1]*LON_RANGE + LON_MIN
#     plt.plot(v_lon,v_lat,linewidth=0.8)
    plt.plot(v_lon,v_lat,color=c,linewidth=0.8)

## Coastlines
if "bretagne" in config.output_filepath:
    for point in l_coastline_poly:
        poly = np.array(point)
        plt.plot(poly[:,0],poly[:,1],color="k",linewidth=0.8)

plt.xlim([LON_MIN,LON_MAX])
plt.ylim([LAT_MIN,LAT_MAX])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(config.output_filepath.replace(".pkl",".png"))