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
                     n_networks=7, n_rois=100, resolution_mm=1, 
                     plot=False, save=True, 
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
                            yeo_networks=n_networks,
                            resolution_mm=resolution_mm)

        # Get coordinates from github
        url = 'https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/' \
                'Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/' \
                'Schaefer2018_'+str(n_rois)+'Parcels_'+str(n_networks)+'Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'
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

sub_path = '/media/dragana.manasova/UNTITLED/Integration/Integration_data_preprocessed_denoised/2021_08_04_INTEGRATION_S06YM'

anat_path = f'{sub_path}/S44_T1w'
anat_name = 'wmv_INTEGRATION_S06YM_S44_T1w.nii' #'mv_INTEGRATION_S06YM_S44_T1w.nii' # wmv_INTEGRATION_S06YM_S44_T1w.nii
anat_file = f"{anat_path}/{anat_name}"

# Read the metadata
df = pd.read_csv('data/metadata_subject_06.csv')
df_run1 = df[(df['block'] == 1) & (df['SNR'] == 'SNR07')]
df_run1 = df[(df['block'] == 1) & (df['status'] == 'HIT')]
df_run1 = df[df['SNR'].isin(['SNR07', 'SNR09'])]
df_sub6 = df[df['status'] == 'HIT']

baseline_correction = True

ts_start = -3
ts_end = 11

all_time_series = []
cort_all_time_series = []
cort_all_separated_time_series = []

for i in range(1,7):
    print(i)
    run_name = f'S{5*(i+2)}_INTEGRATION_run0{i}/denoised/denoised_run0{i}.nii'
    print(f'run_name: {run_name}')
    nii_path = f"{sub_path}/{run_name}"
    img = nib.load(nii_path)

    cort_roi_time_series, cort_roi_labels, cort_roi_coordinates = get_parcellation(sub='sub-06', 
                                                                    sub_ses=f'run-0{i}', 
                                                                    anat_file=anat_file, 
                                                                    bold_file=nii_path, 
                                                                    atlas_name='Schaefer2018', 
                                                                    base_path=derivatives_path, 
                                                                    plot_dir=derivatives_path, 
                                                                    n_networks=17,
                                                                    n_rois=400, 
                                                                    resolution_mm=1, 
                                                                    plot=True, 
                                                                    save=False, 
                                                                    filtered=None # 'filtered'
                                                                    )
    cort_roi_labels = [row.tobytes().decode('UTF-8') for row in cort_roi_labels]
    

    subcort_roi_time_series, subcort_roi_labels, subcort_roi_coordinates = get_parcellation(sub='sub-06', 
                                                                    sub_ses=f'run-0{i}', 
                                                                    anat_file=anat_file, 
                                                                    bold_file=nii_path, 
                                                                    atlas_name='HarvardOxfordSubcortical', 
                                                                    base_path=derivatives_path, 
                                                                    plot_dir=derivatives_path, 
                                                                    plot=True, 
                                                                    save=False, 
                                                                    filtered=None # 'filtered'
                                                                    )
    
    # Get the events from the metadata from the given run
    df_run = df[(df['block'] == i) & (df['status'] == 'HIT')]
    print(f'Number of events: {len(df_run.volume_stim)}')

    fig, axs = plt.subplots(len(subcort_roi_labels), 1, figsize=(20, 10))
    for j, label in enumerate(subcort_roi_labels):
        axs[j].plot(subcort_roi_time_series[:, j])
        axs[j].set_title(f'Run {i}: {label}', pad=-150)
        axs[j].set_ylabel('Signal')
        # Plot vertical lines at the timepoints from df_run01.volume_stim
        for timepoint in df_run.volume_stim:
            axs[j].axvline(x=timepoint, color='r', linestyle='--')
        if j < len(subcort_roi_labels) - 1:  # if not the last subplot
            axs[j].set_xticklabels([])
        else:
            axs[j].set_xlabel('Time points')
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()

    all_regions_time_series = []

    # Loop over each subcortical region
    for region_index, region_label in enumerate(subcort_roi_labels):
        print(f'Processing region {region_label}')

        # Initialize an empty list to store the selected time series
        selected_time_series = []

        # Select the time series of the current region
        region_time_series = subcort_roi_time_series[:, region_index]

        # Loop over each timepoint in df_run1.volume_stim
        for timepoint in df_run.volume_stim:
            cut_region_time_series = region_time_series[timepoint+ts_start:timepoint+ts_end]
            # There will be 7 points before the timepoint
            if baseline_correction: 
                # Compute the average of the first 6 points
                baseline = np.mean(cut_region_time_series[:3]) # [:6] for the -7 to -1 points
                # Subtract the baseline from the time series
                cut_region_time_series = cut_region_time_series - baseline
            selected_time_series.append(cut_region_time_series)

            # Convert the list of selected time series into a matrix
            selected_time_series_matrix = np.array(selected_time_series)
        
        all_regions_time_series.append(selected_time_series_matrix)

        all_regions_time_series_matrix = np.array(all_regions_time_series)

    all_time_series.append(all_regions_time_series_matrix) # 2 blocks len x x number of events 
    

    ##### Cortical regions
    # The difference with the subcortical code above is that
    # I save it for all regions together, and then average them

    # cort_all_regions_time_series = []
    # Initialize an empty list to store the selected time series
    cort_selected_time_series = []

    # Loop over each cortical region
    for region_index, region_label in enumerate(cort_roi_labels):
        print(f'Processing region {region_label}')

        # Select the time series of the current region
        region_time_series = cort_roi_time_series[:, region_index]

        # Loop over each timepoint in df_run.volume_stim
        for timepoint in df_run.volume_stim:
            cut_region_time_series = region_time_series[timepoint+ts_start:timepoint+ts_end]
            # There will be 7 points before the timepoint
            if baseline_correction: 
                # Compute the average of the first 6 points
                baseline = np.mean(cut_region_time_series[:3]) # [:6] for the -7 to -1 points
                # Subtract the baseline from the time series
                cut_region_time_series = cut_region_time_series - baseline
            cort_selected_time_series.append(cut_region_time_series)

            # Convert the list of selected time series into a matrix
            cort_selected_time_series_matrix = np.array(cort_selected_time_series)
    cort_all_time_series.append(cort_selected_time_series_matrix) # 2 blocks len x x number of events

    # In this case below it's also the cortical regions but separated
    # So we get the signals separately per region

    cort_all_regions_time_series = []

    # Loop over each cortical region
    for region_index, region_label in enumerate(cort_roi_labels):
        print(f'Processing region {region_label}')

        cort_selected_time_series = []

        # Select the time series of the current region
        region_time_series = cort_roi_time_series[:, region_index]

        # Loop over each timepoint in df_run.volume_stim
        for timepoint in df_run.volume_stim:
            cut_region_time_series = region_time_series[timepoint+ts_start:timepoint+ts_end]
            # There will be 7 points before the timepoint
            if baseline_correction: 
                # Compute the average of the first 6 points
                baseline = np.mean(cut_region_time_series[:3]) # [:6] for the -7 to -1 points
                # Subtract the baseline from the time series
                cut_region_time_series = cut_region_time_series - baseline
            cort_selected_time_series.append(cut_region_time_series)

            # Convert the list of selected time series into a matrix
            cort_selected_time_series_matrix = np.array(cort_selected_time_series)
        
        cort_all_regions_time_series.append(cort_selected_time_series_matrix)

        cort_all_regions_time_series_matrix = np.array(cort_all_regions_time_series)

    cort_all_separated_time_series.append(cort_all_regions_time_series_matrix) # 2 blocks len x x number of events

    
#%%
"""
selected_time_series_matrix.shape = n_trials x n_times
n_trials is variable = len(df_run.volume_stim)
n_times = 14

len(all_regions_time_series) = 15
15 subcortical regions

all_regions_time_series_matrix.shape = n_regions x n_trials x n_times
n_regions = 15

len(all_time_series) = 6
6 blocks

"""


#%%
# selected_indices = [43, 44, 48, 49, 243, 244, 248, 250]
# selected_indices = [44, 45, 49, 50, 244, 245, 249, 251]
# selected_labels = [cort_roi_labels[i] for i in selected_indices]
# print(selected_labels)

selected_indices = [i for i, label in enumerate(cort_roi_labels) if 'Aud' in label] # + [250] accordin to the paper in todos

selected_labels = [cort_roi_labels[i] for i in selected_indices]
# ['17Networks_LH_SomMotB_Aud_1',
#  '17Networks_LH_SomMotB_Aud_2',
#  '17Networks_LH_SomMotB_Aud_3',
#  '17Networks_LH_SomMotB_Aud_4',
#  '17Networks_RH_SomMotB_Aud_1',
#  '17Networks_RH_SomMotB_Aud_2',
#  '17Networks_RH_SomMotB_Aud_3']

atlas = datasets.fetch_atlas_schaefer_2018(
                    n_rois=400, 
                    yeo_networks=17,
                    resolution_mm=1)
atlas_filename = atlas.maps


# Load the atlas image
atlas_img = nib.load(atlas_filename)

# Get the atlas data
atlas_data = atlas_img.get_fdata()

# Create a mask that only includes the selected regions
mask_data = np.isin(atlas_data, selected_indices)

# Create a new Nifti image from the mask
mask_img = nib.Nifti1Image(mask_data.astype(int), atlas_img.affine)

# Plot the mask
plotting.plot_roi(mask_img, bg_img=atlas_img, display_mode='ortho', cmap='Paired')

# Get the timeseries only for the selected regions in selected_indices
cort_all_separated_time_series_auditory = [elem[selected_indices, :, :] for elem in cort_all_separated_time_series if elem.ndim == 3 and elem.shape[0] >= max(selected_indices)+1]



#%%
# Save the a1ll_time_series
all_time_series_name = os.path.join(derivatives_path, 
        f'func_roi_timeseries/sub-06_run-01-06_atlas-HarvardOxfordSubcortical_all_time_series.npy')

#%%

# subcort_roi_labels :
# ['Left Thalamus',
#  'Left Caudate',
#  'Left Putamen',
#  'Left Pallidum',
#  'Brain-Stem',
#  'Left Hippocampus',
#  'Left Amygdala',
#  'Left Accumbens',
#  'Right Thalamus',
#  'Right Caudate',
#  'Right Putamen',
#  'Right Pallidum',
#  'Right Hippocampus',
#  'Right Amygdala',
#  'Right Accumbens']

#%% Subcortical regions signals plot


# Create a figure with multiple subplots
fig, axs = plt.subplots(5, 3, figsize=(15, 10))

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Generate x values starting from -7
x_values = np.arange(ts_start, len(all_time_series[0][0]) + ts_start)

# Loop over each label in subcort_roi_labels
for i, (ax, roi_name) in enumerate(zip(axs, subcort_roi_labels)):
    # Get the signal corresponding to the current label
    roi_signal = [matrix[i] for matrix in all_time_series]

    # Stack the signals along the second dimension
    stacked_roi_signal = np.vstack(roi_signal)

    # Compute the average of the stacked time series
    average_roi_signal = np.mean(stacked_roi_signal, axis=0)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Plot the average signal with the new x values
    ax.plot(x_values, average_roi_signal, linewidth=2, color='black', 
            label=f'{roi_name}')

    # Plot all the individual signals with the new x values
    for signal in stacked_roi_signal:
        ax.plot(x_values, signal, alpha=0.1)

    # Set the x-ticks to start from ts_start
    ax.set_xticks(x_values)

    # Set the y-axis limits
    ax.set_ylim(-1, 1)
    # Set the y-ticks
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    # Add a vertical dashed line at x=0
    ax.axvline(x=0, color='violet', linestyle='--', linewidth=1)

    ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
# %% All cortical regions together

# There are 73 trials in sub06, 400 ROIs
# so we have 73 x 400 = 29200 time series
# and we average across all of them.

# Another option is to average across the 400ROI, and then average again all trials

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(5, 3))

# Generate x values starting from -7
x_values = np.arange(ts_start, len(cort_all_time_series[0][0]) + ts_start)

# Stack the signals along the second dimension
stacked_cort_signal = np.vstack(cort_all_time_series)

# Compute the average of the stacked time series
average_cort_signal = np.mean(stacked_cort_signal, axis=0)

# Add a horizontal line at y=0
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Plot the average signal with the new x values
ax.plot(x_values, average_cort_signal, linewidth=2, color='black', 
        label='Cortex')

# Plot all the individual signals with the new x values
# for signal in stacked_cort_signal:
#     ax.plot(x_values, signal, alpha=0.1)

# Set the x-ticks to start from ts_start
ax.set_xticks(x_values)

# Set the y-axis limits
ax.set_ylim(-1, 1)
# Set the y-ticks
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

# Add a vertical dashed line at x=0
ax.axvline(x=0, color='violet', linestyle='--', linewidth=1)

ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
# %% Auditory cortical regions

# Create a figure with multiple subplots
fig, axs = plt.subplots(7, 1, figsize=(5, 10))

# Generate x values starting from -7
x_values = np.arange(ts_start, len(cort_all_separated_time_series_auditory[0][0]) + ts_start)

# Loop over each label in selected_labels
for i, (ax, roi_name) in enumerate(zip(axs, selected_labels)):
    # Get the signal corresponding to the current label
    roi_signal = [matrix[i] for matrix in cort_all_separated_time_series_auditory]

    # Stack the signals along the second dimension
    stacked_roi_signal = np.vstack(roi_signal)

    # Compute the average of the stacked time series
    average_roi_signal = np.mean(stacked_roi_signal, axis=0)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Plot the average signal with the new x values
    ax.plot(x_values, average_roi_signal, linewidth=2, color='black', 
            label=f'{roi_name}')

    # Plot all the individual signals with the new x values
    for signal in stacked_roi_signal:
        ax.plot(x_values, signal, alpha=0.1)

    # Set the x-ticks to start from ts_start
    ax.set_xticks(x_values)

    # Set the y-axis limits
    ax.set_ylim(-1, 1)
    # Set the y-ticks
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    # Add a vertical dashed line at x=0
    ax.axvline(x=0, color='violet', linestyle='--', linewidth=1)

    ax.legend(loc='lower right')

plt.tight_layout()
plt.show()


# %% Average auditory cortical regions

fig, ax = plt.subplots(figsize=(5, 2))

# Generate x values starting from -7
x_values = np.arange(ts_start, len(cort_all_separated_time_series_auditory[0][0]) + ts_start)

# Initialize an empty list to store all signals
all_signals = []

# Loop over each label in selected_labels
for i, roi_name in enumerate(selected_labels):
    # Get the signal corresponding to the current label
    roi_signal = [matrix[i] for matrix in cort_all_separated_time_series_auditory]

    # Stack the signals along the second dimension
    stacked_roi_signal = np.vstack(roi_signal)

    # Add the stacked signals to the list of all signals
    all_signals.append(stacked_roi_signal)

# Stack all signals along the first dimension
all_signals_stacked = np.vstack(all_signals)

average_signal = np.mean(all_signals_stacked, axis=0)

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

ax.plot(x_values, average_signal, linewidth=2, color='black')

ax.set_xticks(x_values)

ax.set_ylim(-1, 1)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

ax.axvline(x=0, color='violet', linestyle='--', linewidth=1)

ax.set_title('Average auditory cortical regions')

ax.set_ylabel('Signal change [%]')
ax.set_xlabel('Time [s] (0 = stimulus onset)')

ax.fill_betweenx(ax.get_ylim(), -3, -1, color='lightgray', alpha=0.5)

plt.tight_layout()
plt.show()

# %%
