import os
from pathlib import Path
import pickle

import scipy.signal as spsig
import numpy as np
import matplotlib.pyplot as plt

# # i cant be bothered to get the import working properly right now
# import sys
# sys.path.append(str(Path('.').resolve()))

from lib_mic_array.script.cic.cic_filter import cic_impulse, half_band_calc_filter

DEFAULT_FILTERS_FILE = Path(__file__).parents[2] / "tests" / "signal" / "BasicMicArray" / "default_filters.pkl"

# load the default filters from the pkl file. 
with open(DEFAULT_FILTERS_FILE, "rb") as filt_file:
    stage1, stage2 = pickle.load(filt_file)

s1_coef, s1_df = stage1
s2_coef, s2_df = stage2

fs_target = 16_000
fs_mid = fs_target * s2_df
fs_start = fs_mid * s1_df

# get PDM files
hydra_audio_path = os.environ.get('hydra_audio_PATH', '~/hydra_audio')
pdm_path = Path(hydra_audio_path, "acoustic_team_test_audio", "mic_pdm", "pdm_9k_10s.npy")
pdm_signal = np.load(pdm_path)

pdm_signal = 1 - 2 * pdm_signal # i think this is how you are suppose to interpret the binary pdm

# filter with current design
x = pdm_signal.copy()
x = spsig.lfilter(s1_coef, [1], x)[::s1_df]
x = spsig.lfilter(s2_coef, [1], x)[::s2_df]
y_current = x.copy()

# prototype filters
s1_coef = cic_impulse(s1_df, 4)

# split the 2nd stage
s3_df = s2_df // 2
s2_df = 2

f_pass = 6300
s2_coef = half_band_calc_filter(fs_mid, f_pass, 30)
s3_coef = spsig.firwin(31, f_pass, window=('kaiser', 4), fs=fs_mid//s2_df)

x = pdm_signal.copy()
x = spsig.lfilter(s1_coef, [1], x)[::s1_df]
x = spsig.lfilter(s2_coef, [1], x)[::s2_df]
x = spsig.lfilter(s3_coef, [1], x)[::s3_df]
y_prototype = x.copy()

f, S_pdm = spsig.welch(pdm_signal, fs=fs_start, window='blackmanharris', nperseg=(s1_df * s2_df * s3_df) * 4096, scaling='spectrum')
plt.plot(f, 10 * np.log10(S_pdm), label='PDM')

f, S_current = spsig.welch(y_current, fs=fs_target, window='blackmanharris', nperseg=4096, scaling='spectrum')
f, S_prototype = spsig.welch(y_prototype, fs=fs_target, window='blackmanharris', nperseg=4096, scaling='spectrum')

plt.plot(f, 10 * np.log10(S_current), label='current')
plt.plot(f, 10 * np.log10(S_prototype), label='prototype')
plt.xlim(f[0], f[-1])
plt.legend()

plt.savefig('tmp_cic_prototype_decimator.png')
