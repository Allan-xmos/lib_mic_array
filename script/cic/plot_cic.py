import os
from pathlib import Path
import pickle

import scipy.signal as spsig
import numpy as np
import matplotlib.pyplot as plt

# # i cant be bothered to get the import working properly right now
# import sys
# sys.path.append(str(Path('.').resolve()))

from lib_mic_array.script.cic.cic_filter import cic_ma, cic, cic_impulse

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
pdm_path = Path(hydra_audio_path, "acoustic_team_test_audio", "mic_pdm", "pdm_1k_10s.npy")
pdm_signal = np.load(pdm_path)

pdm_signal = 1 - 2 * pdm_signal # i think this is how you are suppose to interpret the binary pdm

fg, ax = plt.subplots(2)
f, Sxx = spsig.welch(pdm_signal, fs=fs_start, window='blackmanharris', nperseg=s1_df*4096, scaling='spectrum')
ax[1].plot(f, 10 * np.log10(Sxx), label='pdm')
for order in [1, 4]:
    y_ma = cic_ma(pdm_signal, s1_df, order)

    #use optimised code
    y_cic = cic(pdm_signal, s1_df, order)

    ax[0].plot(y_ma)
    ax[0].plot(y_cic)

    f, Sma = spsig.welch(y_ma, fs=fs_mid, window='blackmanharris', nperseg=4096, scaling='spectrum')
    f, Scic = spsig.welch(y_cic, fs=fs_mid, window='blackmanharris', nperseg=4096, scaling='spectrum')

    ax[1].plot(f, 10 * np.log10(Sma), label=f'ma {order}')
    ax[1].plot(f, 10 * np.log10(Scic), label=f'cic {order}')

ax[1].set_xlim(f[0], f[-1])
plt.legend()

plt.savefig('tmp_cic_ma_vs_true.png')

fg, ax = plt.subplots(1)
worN = 4096
f, H = spsig.freqz(s1_coef, [1], worN=worN, fs=fs_start)
ax.plot(f, 20 * np.log10(np.abs(H)), label='S1 lib_mic_array')
for order in [1, 2, 3, 4]:
    b = cic_impulse(s1_df, order)
    f, H = spsig.freqz(b, [1], worN=worN, fs=fs_start)
    ax.plot(f, 20 * np.log10(np.abs(H)), label=f'cic order {order}')

plt.legend()
plt.savefig('tmp_cic_vs_s1_lib_mic_array.png')