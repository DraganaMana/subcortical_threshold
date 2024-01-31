#%% .mat file, outdated, Basak sent me a csv file
from scipy.io import loadmat
import pandas as pd
import numpy as np

eves = loadmat('data/subject_06.mat')

df = pd.DataFrame({key: pd.Series(value[0]) for key, value in eves.items() if isinstance(value, np.ndarray) and value.size > 0})

# remove the rows that have NaN values in both the real_timings and the block_duration columns
# df = df.dropna(subset=['Real_timings', 'block_duration'], how='all')

# save the df as csv
df.to_csv('data/subject_06.csv', index=False)
# %% read the csv file
import pandas as pd
import numpy as np

df = pd.read_csv('data/metadata_subject_06.csv')

# get a subdataframe with block=1, SNR= SNR07
df_run1 = df[(df['block'] == 1) & (df['SNR'] == 'SNR07')]
