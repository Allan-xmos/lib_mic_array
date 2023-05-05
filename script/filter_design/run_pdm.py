import os
from pathlib import Path

import scipy.signal as spsig
import numpy as np
import matplotlib.pyplot as plt


def process_pdm(pdm_signal, coeff_1, coeff_2, decimation_1, decimation_2):
    noise = pdm_signal.astype(np.float64)
    noise = spsig.resample_poly(noise, 1, decimation_1, window=coeff_1)
    noise = spsig.resample_poly(noise, 1, decimation_2, window=coeff_2)

    return noise


if __name__ == "__main__":
    # load default coefficients & decimation factors
    coeffs = np.load(Path(Path(__file__).parent, "..", "..", "tests", "signal", "BasicMicArray", "default_filters.pkl"),
                     allow_pickle=True)

    coeff_1 = coeffs[0][0]
    coeff_2 = coeffs[1][0]

    # define sample rates of each stage
    fs_0 = 3072000

    decimation_1 = coeffs[0][1]
    decimation_2 = coeffs[1][1]

    fs_1 = fs_0 / decimation_1
    fs_2 = fs_1 / decimation_2

    # get PDM files
    hydra_audio_path = os.environ.get('hydra_audio_PATH', '~/hydra_audio')
    pdm_path = Path(hydra_audio_path, "acoustic_team_test_audio", "mic_pdm", "pdm_1k_10s.npy")
    pdm_signal = np.load(pdm_path, allow_pickle=True)

    # run and plot
    output = process_pdm(pdm_signal, coeff_1, coeff_2, decimation_1, decimation_2)

    plt.figure()
    plt.psd(output, Fs=fs_2, NFFT=1024, label='lib_mic_array coeffs')
    plt.legend()
    plt.title("PDM conversion, %s" % pdm_path.name)

    plt.figure()
    plt.specgram(output, Fs=fs_2)
    plt.title("PDM conversion, %s" % pdm_path.name)

    plt.show()
