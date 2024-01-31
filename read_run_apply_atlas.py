#%% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

import nibabel as nib
from nilearn import plotting
from nilearn import image
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker

#%% Functions

def remove_elements(elements_to_remove, roi_names, time_series):
    # Get the indices of the elements to remove
    indices_to_remove = [i for i, x in enumerate(roi_names) if x in elements_to_remove]

    # Remove the elements from the list
    for index in sorted(indices_to_remove, reverse=True):
        del roi_names[index]
    
    # Remove the columns with the same index as the removed elements
    time_series = np.delete(time_series, indices_to_remove, axis=1)
    
    return roi_names, time_series

def get_parcellation(sub, sub_ses, anat_file, bold_file, atlas_name, 
                     base_path, plot_dir, 
                     n_rois=100, resolution_mm=1, plot=False, save=True, 
                     filtered='filtered'):
    """

    Input : 
        bold path : 
        atlas : Schaefer2018, AAL
        n_rois : (int) Number of ROIs we walk to get from the parcellations. 
        resolution_mm : (int) 1 or 2, the resolution in mm of the parcellation
        plot : (bool) True if you want to plot and save the plots

    Output : 
        roi_coordinates : (list of lists) n_rois * 3 
    """
    if atlas_name == 'Schaefer2018':
        atlas = datasets.fetch_atlas_schaefer_2018(
                            n_rois=n_rois, 
                            yeo_networks=7,
                            resolution_mm=resolution_mm)

        # Get coordinates from github
        url = 'https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/' \
                'Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/' \
                'Schaefer2018_'+str(n_rois)+'Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'
        schaefer_coordinates = pd.read_csv(url)
        roi_coordinates = schaefer_coordinates[['R', 'A', 'S']].values.tolist() 
        # matching the format of the nilearn other maps
    elif atlas_name == 'AAL':
        print('Attention : Labels not verified for AAL atlas !!!')
        atlas = datasets.fetch_atlas_aal()
        roi_coordinates = None
    elif atlas_name == 'HarvardOxfordSubcortical':
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
        roi_coordinates = None
    else:
        raise ValueError('atlas_name should be Schaefer2018, AAL or HarvardOxfordSubcortical')
    
    # Get the atlas details 
    atlas_filename = atlas.maps
    roi_labels = atlas.labels

    if atlas == 'Schaefer2018':
        roi_labels = [row.tobytes().decode('UTF-8') for row in roi_labels]
        # Check that the Schaefer coordinates from github are ordered the same 
        # as the labels in the nilearn atlas
        rois_sum_check = np.sum(schaefer_coordinates['ROI Name'].to_numpy()==roi_labels)  
        # if this sum is equal to the n_rois, it means all the labels are the same
        if rois_sum_check != n_rois:
            print(f'\n!!!!!!\n n_rois check failed. rois_sum_check={rois_sum_check} and n_rois={n_rois}')
    
    # Create a masker object to extract the timeseries from the atlas
    # masker = NiftiLabelsMasker(labels_img=atlas_filename,
    #                             labels=roi_labels,
    #                             resampling_target='labels',
    #                             standardize=False, 
    #                             verbose=5)
    masker = NiftiLabelsMasker(labels_img=atlas_filename,
                            standardize=False, 
                            verbose=5
                            )
    roi_time_series = masker.fit_transform(bold_file) # [bold volumes, n_rois]
    print('roi_time_series shape after masker: ', roi_time_series.shape)

    if atlas_name=='HarvardOxfordSubcortical':
        roi_labels = roi_labels[1:]
        print('We have the following labels : ', roi_labels)
        roi_to_remove = ['Left Cerebral White Matter',
                 'Left Cerebral Cortex ',
                 'Left Lateral Ventrical', 
                 'Right Cerebral White Matter',
                 'Right Cerebral Cortex ',
                 'Right Lateral Ventricle']
        print('Removing the following labels : ', roi_to_remove)
        roi_labels, roi_time_series = remove_elements(roi_to_remove, roi_labels, roi_time_series)
    n_rois = len(roi_labels)
    # Check that the second dimension of the roi_time_series is equal to the number of ROI labels
    if roi_time_series.shape[1] != n_rois:
        raise ValueError(f'roi_time_series.shape[1] = {roi_time_series.shape[1]} and n_rois = {n_rois}. They should be equal.')
        # print(f'roi_time_series.shape[1] = {roi_time_series.shape[1]} and n_rois = {n_rois}. They should be equal.')
    
    if save:
        # Save the roi time series and labels
        roi_time_series_name = os.path.join(base_path, 
                f'00_derivatives/func_roi_timeseries/{filtered}/{atlas_name}/{sub}_{sub_ses}_atlas-{atlas_name}_{len(roi_labels)}_roi-time-series.npy')
        roi_labels_name = os.path.join(base_path,
                f'00_derivatives/func_roi_timeseries/{filtered}/{atlas_name}/{sub}_{sub_ses}_atlas-{atlas_name}_{len(roi_labels)}_roi-labels.npy')

        print(f'Saving roi time series and labels at : \n \
                {roi_time_series_name} \nand\n {roi_labels_name}')
        np.save(roi_time_series_name, roi_time_series, allow_pickle=True)
        np.save(roi_labels_name, roi_labels, allow_pickle=True)

    if plot: 
        # Plot the atlas on the anatomical image

        plotting.plot_roi(atlas_filename, bg_img=anat_file)
        if save:
            plot_folder = os.path.join(base_path, plot_dir, str(date.today()))
            os.makedirs(plot_folder, exist_ok=True)
            plt.savefig(os.path.join(plot_folder, f'{sub}_{sub_ses}_{atlas_name}_{str(n_rois)}_rois.png'), 
                        bbox_inches='tight')
            plt.close()
    return roi_time_series, roi_labels, roi_coordinates


#%% Main

local_base_path = '/home/dragana.manasova/Documents/codes/subcortical_integration'
derivatives_path = os.path.join(local_base_path, 'derivatives')


#%% Reading and atlasing the func file
func_path = '/media/dragana.manasova/UNTITLED/Integration/Integration_data_preprocessed_denoised/2021_08_04_INTEGRATION_S06YM/S15_INTEGRATION_run01/denoised'
run_name = 'denoised_run01.nii'

nii_path = f"{func_path}/{run_name}"
img = nib.load(nii_path)

#%% Concatenate the 6 runs together

sub_path = '/media/dragana.manasova/UNTITLED/Integration/Integration_data_preprocessed_denoised/2021_08_04_INTEGRATION_S06YM'
# 3 4 5 6 7 8
# 1+2 2+2 3+2 4+2 5+2 6+2


for i in range(1,7):
    print(i)
    run_name = f'S{5*(i+2)}_INTEGRATION_run0{i}/denoised/denoised_run0{i}.nii'
    print(f'run_name: {run_name}')
    nii_path = f"{sub_path}/{run_name}"
    img = nib.load(nii_path)
    if i == 1:
        img_concat = img
    else:
        img_concat = image.concat_imgs([img_concat, img])


#%%
"""
# Load the atlas
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)

# Resample the atlas to match the .nii file
resampled_atlas = image.resample_to_img(atlas.maps, img)

# Initialize an empty list to store the labeled images
labeled_imgs = []

# Iterate over the fourth dimension of img
for i in range(img.shape[3]):
    # Select the image at the current time point
    img_i = image.index_img(img, i)
    
    # Apply the atlas
    labeled_img_i = image.math_img("img1 * img2", img1=img_i, img2=resampled_atlas)
    
    # Add the labeled image to the list
    labeled_imgs.append(labeled_img_i)

# Concatenate the labeled images along the fourth dimension
labeled_img = image.concat_imgs(labeled_imgs)
"""

# %% Reading the anatomical file & plotting the func on the anat

anat_path = '/media/dragana.manasova/UNTITLED/Integration/Integration_data_preprocessed_denoised/2021_08_04_INTEGRATION_S06YM/S44_T1w'
anat_name = 'wmv_INTEGRATION_S06YM_S44_T1w.nii' #'mv_INTEGRATION_S06YM_S44_T1w.nii' # wmv_INTEGRATION_S06YM_S44_T1w.nii

anat_file = f"{anat_path}/{anat_name}"

#%%
"""
# Select one timepoint from the 4D labeled_img
labeled_img_timepoint = image.index_img(labeled_img, 1)

# Plot the labeled image with the given atlas
plotting.plot_roi(labeled_img_timepoint, bg_img=anat_file)

plotting.plot_roi(labeled_img_timepoint, bg_img=atlas.maps)
"""

# %%

cort_roi_time_series, cort_roi_labels, cort_roi_coordinates = get_parcellation(sub='sub-06', 
                                                                sub_ses='run-01', 
                                                                anat_file=anat_file, 
                                                                bold_file=nii_path, 
                                                                atlas_name='Schafer2018', 
                                                                base_path=derivatives_path, 
                                                                plot_dir=derivatives_path, 
                                                                n_rois=100, 
                                                                resolution_mm=1, 
                                                                plot=True, 
                                                                save=False, 
                                                                filtered=None # 'filtered'
                                                                )

subcort_roi_time_series, subcort_roi_labels, subcort_roi_coordinates = get_parcellation(sub='sub-06', 
                                                                sub_ses='run-01', 
                                                                anat_file=anat_file, 
                                                                bold_file=nii_path, 
                                                                atlas_name='HarvardOxfordSubcortical', 
                                                                base_path=derivatives_path, 
                                                                plot_dir=derivatives_path, 
                                                                n_rois=100, 
                                                                resolution_mm=1, 
                                                                plot=True, 
                                                                save=False, 
                                                                filtered=None # 'filtered'
                                                                )

#%%
# Subcortical for the concatenated nii f-image
subcort_roi_time_series, subcort_roi_labels, subcort_roi_coordinates = get_parcellation(sub='sub-06', 
                                                                sub_ses='run-01', 
                                                                anat_file=anat_file, 
                                                                bold_file=img_concat, 
                                                                atlas_name='HarvardOxfordSubcortical', 
                                                                base_path=derivatives_path, 
                                                                plot_dir=derivatives_path, 
                                                                n_rois=100, 
                                                                resolution_mm=1, 
                                                                plot=True, 
                                                                save=False, 
                                                                filtered=None # 'filtered'
                                                                )
# %%


fig, axs = plt.subplots(len(subcort_roi_labels), 1, figsize=(20, 10))
for i, label in enumerate(subcort_roi_labels):
    axs[i].plot(subcort_roi_time_series[:, i])
    axs[i].set_title(label, pad=-150)
    axs[i].set_ylabel('Signal')
    if i < len(subcort_roi_labels) - 1:  # if not the last subplot
        axs[i].set_xticklabels([])
    else:
        axs[i].set_xlabel('Time points')
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()

#%%

# Read the metadata
df = pd.read_csv('data/metadata_subject_06.csv')
df_run1 = df[(df['block'] == 1) & (df['SNR'] == 'SNR07')]

df_run1 = df[(df['block'] == 1) & (df['status'] == 'HIT')]

# get both the SNR07 and SNR09:
df_run1 = df[df['SNR'].isin(['SNR07', 'SNR09'])]

df_sub6 = df[df['status'] == 'HIT']



# %%
fig, axs = plt.subplots(len(subcort_roi_labels), 1, figsize=(20, 10))
for i, label in enumerate(subcort_roi_labels):
    axs[i].plot(subcort_roi_time_series[:, i])
    axs[i].set_title(label, pad=-150)
    axs[i].set_ylabel('Signal')
    # Plot vertical lines at the timepoints from df_run01.volume_stim
    for timepoint in df_sub6.volume_stim_global:
        axs[i].axvline(x=timepoint, color='r', linestyle='--')
    if i < len(subcort_roi_labels) - 1:  # if not the last subplot
        axs[i].set_xticklabels([])
    else:
        axs[i].set_xlabel('Time points')
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()
# %%

# Loop over each subcortical region
for region_index, region_label in enumerate(subcort_roi_labels):
    # Select the time series of the current region
    region_time_series = subcort_roi_time_series[:, region_index]

    # Initialize an empty list to store the selected time series
    selected_time_series = []

    # Loop over each timepoint in df_run1.volume_stim
    for timepoint in df_run1.volume_stim:
        # Select the time series at the current timepoint and the following 5 timepoints
        selected_time_series.append(region_time_series[timepoint:timepoint+11])

    # Convert the list of selected time series into a matrix
    selected_time_series_matrix = np.array(selected_time_series)

    # Compute the average of the matrix
    average_time_series = np.mean(selected_time_series_matrix, axis=0)

    # Plot the average time series
    plt.plot(average_time_series, linewidth=2, label=region_label)

    # Plot all the individual time series
    for i in range(selected_time_series_matrix.shape[0]):
        plt.plot(selected_time_series_matrix[i], alpha=0.5)

    plt.legend()
    plt.show()

#%% 
# Create a figure with multiple subplots
fig, axs = plt.subplots(len(subcort_roi_labels), 1, figsize=(5, len(subcort_roi_labels)*2))

# Loop over each subcortical region
for region_index, region_label in enumerate(subcort_roi_labels):
    # Select the time series of the current region
    region_time_series = subcort_roi_time_series[:, region_index]

    # Initialize an empty list to store the selected time series
    selected_time_series = []

    # Loop over each timepoint in df_run1.volume_stim
    for timepoint in df_run1.volume_stim:
        # Select the time series at the current timepoint and the following 5 timepoints
        selected_time_series.append(region_time_series[timepoint:timepoint+11])

    # Convert the list of selected time series into a matrix
    selected_time_series_matrix = np.array(selected_time_series)

    # Compute the average of the matrix
    average_time_series = np.mean(selected_time_series_matrix, axis=0)

    # Plot the average time series in the current subplot
    axs[region_index].plot(average_time_series, linewidth=2, label=region_label)

    # Plot all the individual time series in the current subplot
    for i in range(selected_time_series_matrix.shape[0]):
        axs[region_index].plot(selected_time_series_matrix[i], alpha=0.2)

    axs[region_index].legend(loc='lower right')

plt.tight_layout()
plt.show()
# %%
