import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import sys
import numpy as np
import matplotlib.image as mpimg
from scipy import signal
from matplotlib.widgets import Slider

# Load data from drives
f_damper_defect = h5py.File('../damper/defect/defect_060510.hdf', 'r')
f_damper_intact = h5py.File('../damper/intact/intact_049594.hdf', 'r')

# Number of datapoints per segment
dp = 2048


# DEFECT - Get the necessary channel group keys 
def get_channel_group_keys(f_damper_type):
    result_dict = dict()
    keys_1 = list(f_damper_type.keys())
    for keys_2 in keys_1[1:]:
        for key in f_damper_type[keys_2].keys():
            if key == 'SARA_Accel_X_b':
                result_dict["accel_x"] = f_damper_type[keys_2]['SARA_Accel_X_b']
            elif key == 'SARA_CF_Accel_X_b':
                result_dict["control_accel_x"] = f_damper_type[keys_2]['SARA_CF_Accel_X_b']
            elif key == 'SARA_Accel_Y_b':
                result_dict["accel_y"] = f_damper_type[keys_2]['SARA_Accel_Y_b']
            elif key == 'SARA_CF_Accel_Y_b':
                result_dict["control_accel_y"] = f_damper_type[keys_2]['SARA_CF_Accel_Y_b']
            elif key == 'SARA_Accel_Z_b':
                result_dict["accel_z"] = f_damper_type[keys_2]['SARA_Accel_Z_b']
            elif key == 'SARA_CF_Accel_Z_b':
                result_dict["control_accel_z"] = f_damper_type[keys_2]['SARA_CF_Accel_Z_b']
            elif key == 'ESP_HL_Radgeschw':   
                result_dict["rear_left_velo"] = f_damper_type[keys_2]['ESP_HL_Radgeschw']
            elif key == 'ESP_HR_Radgeschw':   
                result_dict["rear_right_velo"] = f_damper_type[keys_2]['ESP_HR_Radgeschw']
            elif key == 'ESP_VL_Radgeschw':   
                result_dict["front_left_velo"] = f_damper_type[keys_2]['ESP_VL_Radgeschw']
            elif key == 'ESP_VR_Radgeschw':   
                result_dict["front_right_velo"] = f_damper_type[keys_2]['ESP_VR_Radgeschw']
                
    return result_dict

damper_defect_values = get_channel_group_keys(f_damper_defect)
damper_intact_values = get_channel_group_keys(f_damper_intact)

# Delete 'Init' values
def delete_init_from_velocities(damper_type_values):
    rear_left_velo_index = np.where(np.array(damper_type_values["rear_left_velo"]).astype('str') == "Init")
    damper_type_values["rear_left_velo"] = np.delete(damper_type_values["rear_left_velo"], rear_left_velo_index)
    rear_right_velo_index = np.where(np.array(damper_type_values["rear_right_velo"]).astype('str') == "Init")
    damper_type_values["rear_right_velo"] = np.delete(damper_type_values["rear_right_velo"], rear_right_velo_index)
    front_left_velo_index = np.where(np.array(damper_type_values["front_left_velo"]).astype('str') == "Init")
    damper_type_values["front_left_velo"] = np.delete(damper_type_values["front_left_velo"], front_left_velo_index)
    front_right_velo_index = np.where(np.array(damper_type_values["front_right_velo"]).astype('str') == "Init")
    damper_type_values["front_right_velo"] = np.delete(damper_type_values["front_right_velo"], front_right_velo_index)
    
    return damper_type_values

def delete_init_from_accs(damper_type_values):
    acc_x_init_index = np.where(np.array(damper_type_values["control_accel_x"]).astype('str') == "Init")
    damper_type_values["accel_x"] = np.delete(damper_type_values["accel_x"], acc_x_init_index)
    
    acc_y_init_index = np.where(np.array(damper_type_values["control_accel_y"]).astype('str') == "Init")
    damper_type_values["accel_y"] = np.delete(damper_type_values["accel_y"], acc_y_init_index)
    
    acc_z_init_index = np.where(np.array(damper_type_values["control_accel_z"]).astype('str') == "Init")
    damper_type_values["accel_z"] = np.delete(damper_type_values["accel_z"], acc_z_init_index)
    
    return damper_type_values

damper_defect_values = delete_init_from_velocities(damper_defect_values)
damper_intact_values = delete_init_from_velocities(damper_intact_values)
damper_defect_values = delete_init_from_accs(damper_defect_values)
damper_intact_values = delete_init_from_accs(damper_intact_values)


# Calculate the mean velocity
mean_velocity_defect = (np.array(damper_defect_values["rear_left_velo"]).astype(np.float64)
                 + np.array(damper_defect_values["rear_right_velo"]).astype(np.float64)
                 + np.array(damper_defect_values["front_left_velo"]).astype(np.float64)
                 + np.array(damper_defect_values["front_right_velo"]).astype(np.float64)) / 4 

mean_velocity_intact = (np.array(damper_intact_values["rear_left_velo"]).astype(np.float64)
                 + np.array(damper_intact_values["rear_right_velo"]).astype(np.float64)
                 + np.array(damper_intact_values["front_left_velo"]).astype(np.float64)
                 + np.array(damper_intact_values["front_right_velo"]).astype(np.float64)) / 4 

# mean velocity of 512 data points corresponding to one spectrogram image
def mean_velo_for_512_points(mean_velocity_type):
    mean_velocity_512_data_points = []
    i = 0
    while i < mean_velocity_type.shape[0]: # 2345 * 512 = 1200640
        tmp = sum(mean_velocity_type[i:i+512]) / 512
        mean_velocity_512_data_points.append(tmp)
        i += 512

    return np.array(mean_velocity_512_data_points)

mean_velo_values_defect = mean_velo_for_512_points(mean_velocity_defect)
mean_velo_values_intact = mean_velo_for_512_points(mean_velocity_intact)

# Delete corresponding accelerations with less than 30 km/h velocities

def delete_accel_less_than_30_velo(accel_type_str, accel_type_array, mean_velo_values, stop_data_point):
    # indices with less than 30 km/h velocity 
    mean_velocity_indices = np.where(mean_velo_values < 30.0)
    accel_type_array = accel_type_array.astype('double')
    if accel_type_str == "z":
        accel_type_array -= 9.8
    s = int(stop_data_point / dp)
    
    reshaped_accel_type_values = np.reshape(accel_type_array[:s*dp], (s, dp))
    reshaped_accel_type_values_cleaned = np.delete(reshaped_accel_type_values, mean_velocity_indices[0][:-2], axis=0)
    flattened = reshaped_accel_type_values_cleaned.flatten()
    return flattened

# defect acceleration data
accel_x_array_defect = np.array(damper_defect_values["accel_x"])
accel_y_array_defect = np.array(damper_defect_values["accel_y"])
accel_z_array_defect = np.array(damper_defect_values["accel_z"])
# intact acceleration data
accel_x_array_intact = np.array(damper_intact_values["accel_x"])
accel_y_array_intact = np.array(damper_intact_values["accel_y"])
accel_z_array_intact = np.array(damper_intact_values["accel_z"])

min_accel_point_of_xyz_defect = min([accel_x_array_defect.shape[0], accel_y_array_defect.shape[0], accel_z_array_defect.shape[0]])
min_accel_point_of_xyz_intact = min([accel_x_array_intact.shape[0], accel_y_array_intact.shape[0], accel_z_array_intact.shape[0]])

# defect flattened acceleration data
flattened_accel_x_defect = delete_accel_less_than_30_velo("x", accel_x_array_defect, mean_velo_values_defect, min_accel_point_of_xyz_defect)
flattened_accel_y_defect = delete_accel_less_than_30_velo("y", accel_y_array_defect, mean_velo_values_defect, min_accel_point_of_xyz_defect)
flattened_accel_z_defect = delete_accel_less_than_30_velo("z", accel_z_array_defect, mean_velo_values_defect, min_accel_point_of_xyz_defect)
# intact flattened acceleration data
flattened_accel_x_intact = delete_accel_less_than_30_velo("x", accel_x_array_intact, mean_velo_values_intact, min_accel_point_of_xyz_intact)
flattened_accel_y_intact = delete_accel_less_than_30_velo("y", accel_y_array_intact, mean_velo_values_intact, min_accel_point_of_xyz_intact)
flattened_accel_z_intact = delete_accel_less_than_30_velo("z", accel_z_array_intact, mean_velo_values_intact, min_accel_point_of_xyz_intact)


# Draw spectrograms

# Overall spectrogram for damper defect acceleration in Z axis
cmap = plt.get_cmap()
cmap.set_under(color='k', alpha=None)
fig, ax = plt.subplots(figsize=(20,5))
pxx, freq, t, cax = ax.specgram(flattened_accel_z_defect, # first channel
                                Fs=200,                  # to get frequency axis in Hz
                                cmap=cmap)
cbar = fig.colorbar(cax)
cbar.set_label('Intensity dB')
ax.axis("tight")

# Calculate Db power for scaling and restraining min and max
max_db = 10 * np.log10(pxx.max())
min_db = 10 * np.log10(pxx.min())


# Create defect spectrograms
cmap = plt.get_cmap() 
cmap.set_under(color='k', alpha=None)

#for removing numbers on the axes
fig,ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.axis('tight')
ax.axis('off')
#plot spectrogram
i=0
flattened_size_defect = flattened_accel_z_defect.shape[0]
while i < flattened_size_defect: 
    pxx, freq, t, cax_z = ax.specgram(flattened_accel_z_defect[i:i + dp], # first channel
                                    Fs=200,                            # to get frequency axis in Hz
                                    cmap=cmap, vmin=-100, vmax=max_db)
    plt.savefig("../colored_Accel_Z_dataset/defect/" + str(int(i/dp)) + ".png")
    i = i + dp


# Create intact spectrograms
cmap = plt.get_cmap() 
cmap.set_under(color='k', alpha=None)

# for removing numbers on the axes
fig,ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.axis('tight')
ax.axis('off')
# plot spectrogram
i=0
flattened_size_intact = flattened_accel_z_intact.shape[0]
while i < flattened_size_intact:
    pxx, freq, t, cax_z = ax.specgram(flattened_accel_z_intact[i:i + dp], # first channel
                                    Fs=200,                            # to get frequency axis in Hz
                                    cmap=cmap, vmin=-100, vmax=max_db)
    plt.savefig("../colored_Accel_Z_dataset/intact/" + str(int(i/dp)) + ".png")    
    i = i + dp
