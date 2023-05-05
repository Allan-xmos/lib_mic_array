from pathlib import Path

import scipy.signal as spsig
import numpy as np
import matplotlib.pyplot as plt


def plot_filters(coeff_1, coeff_2, fs_0, fs_1, fs_2):

    # calculate combined filter by upsampling the 2nd stage coefficients
    # using the 1st stage filter
    coeff_combo = spsig.resample_poly(coeff_2, 32, 1, window=coeff_1)/32

    # calculate filter frequency responses
    w_1, response_1 = spsig.freqz(coeff_1, fs=fs_0, worN=32768)
    w_2, response_2 = spsig.freqz(coeff_2, fs=fs_1, worN=(32768//32))
    w_combo, response_combo = spsig.freqz(coeff_combo, fs=fs_0, worN=32768)

    fig, axs = plt.subplots(3, 3)

    # stage 1
    axs[0, 0].set_title("stage 1")
    axs[0, 0].plot(coeff_1)
    axs[0, 0].set_xlim([0, len(coeff_1)])
    axs[0, 0].set_ylabel("filter taps")
    axs[1, 0].plot(w_1, 20*np.log10(np.abs(response_1)))
    axs[1, 0].set_xlim([0, w_1[-1]])
    axs[1, 0].set_ylim([-120, 10])
    axs[1, 0].set_ylabel("filter response")
    axs[2, 0].plot(w_1, 20*np.log10(np.abs(response_1)))
    axs[2, 0].set_xlim([0, fs_1/2*1.2])
    axs[2, 0].set_ylim([-10, 10])
    axs[2, 0].set_ylabel("passband ripple")

    # stage 2
    axs[0, 1].set_title("stage 2")
    axs[0, 1].plot(coeff_2)
    axs[0, 1].set_xlim([0, len(coeff_2)])
    axs[1, 1].plot(w_2, 20*np.log10(np.abs(response_2)))
    axs[1, 1].set_xlim([0, w_2[-1]])
    axs[1, 1].set_ylim([-120, 10])
    axs[2, 1].plot(w_2, 20*np.log10(np.abs(response_2)))
    axs[2, 1].set_xlim([0, fs_2/2*1.2])
    axs[2, 1].set_ylim([-10, 10])

    # combined
    axs[0, 2].set_title("combined")
    axs[0, 2].plot(coeff_combo)
    axs[0, 2].set_xlim([0, len(coeff_combo)])
    axs[1, 2].plot(w_combo, 20*np.log10(np.abs(response_combo)))
    axs[1, 2].set_xlim([0, w_combo[-1]])
    axs[1, 2].set_ylim([-120, 10])
    axs[2, 2].plot(w_combo, 20*np.log10(np.abs(response_combo)))
    axs[2, 2].set_xlim([0, fs_2/2*1.2])
    axs[2, 2].set_ylim([-10, 10])

    plt.show()


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

    plot_filters(coeff_1, coeff_2, fs_0, fs_1, fs_2)
