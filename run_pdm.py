import os
from pathlib import Path

import soundfile as sf
import scipy.signal as spsig
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "python", "filter_design"))

import filter_tools as ft
from design_filter import stage_params, design_2_stage, small_768k_to_16k_filter
import plot_coeffs as pc


def process_pdm(pdm_signal, coeff_1, coeff_2, decimation_1, decimation_2):
    noise = pdm_signal.astype(np.float64)
    noise = spsig.resample_poly(noise, 1, decimation_1, window=coeff_1)
    noise = np.concatenate((np.zeros(int(len(coeff_2) - np.ceil(decimation_2))), noise))
    if decimation_2 == 1.5:
        noise = spsig.resample_poly(noise, 2, 2*decimation_2, window=coeff_2)
    else:
        noise = spsig.resample_poly(noise, 1, decimation_2, window=coeff_2)

    return noise


if __name__ == "__main__":
    # load default coefficients & decimation factors
    fs_0 = 3072000//4
    decimations = [32, 1.5]

    # # stage 1 parameters
    # ma_stages = 5

    # # stage 2 parameters
    # cutoff = 5000
    # transition_bandwidth = 0
    # taps_2 = 400
    # fir_window = ("kaiser", 10)
    # stage_2 = stage_params(cutoff, transition_bandwidth, taps_2, fir_window)

    coeffs = small_768k_to_16k_filter(int_coeffs=False)
    coeffs_int = small_768k_to_16k_filter(int_coeffs=True)
    coeffs_int = [ft.normalise_coeffs(np.float64(coeffs_int[0][0])), coeffs_int[0][1]], [ft.normalise_coeffs(np.float64(coeffs_int[1][0])), coeffs_int[1][1]]

    # get PDM files
    pdm_path = Path(r"C:\Users\allanskellett\OneDrive - Xmos\Docs\029_mic_testing\029_mic_testing\768_pdm_xscope.npy")
    pdm_signal = np.load(pdm_path, allow_pickle=True)



    fig, axs =plt.subplots(4, 1, sharex=True)
    axs[0].plot(pdm_signal[:9000])
    axs[0].set_title("PDM signal")
    axs[0].grid(True)

    axs[1].plot(np.abs(np.diff(pdm_signal[:9000])))
    axs[1].set_title("PDM signal diff")
    axs[1].grid(True)

    axs[2].plot(np.cumsum(pdm_signal[:9000]))
    axs[2].set_title("PDM signal cumsum")
    axs[2].grid(True)
    # pdm_signal = pdm_signal[777:]

    # prepend with 256-32 01 to represent initial conditions
    pdm_signal = np.concatenate((np.repeat([0, 1], (256-32)//2, axis=0).astype(np.int8), pdm_signal))

    out_signal = process_pdm(pdm_signal, coeffs[0][0], coeffs[1][0], decimations[0], decimations[1])
    out_signal_int = process_pdm(pdm_signal, coeffs_int[0][0], coeffs_int[1][0], decimations[0], decimations[1])
    fs_1 = fs_0 / decimations[0]
    fs_2 = fs_1 / decimations[1]

    ax0 = pc.plot_stage(coeffs_int[0], fs_0, fs_2)
    ax1 = pc.plot_stage(coeffs_int[1], fs_1, fs_2)
    ax2 = pc.plot_filters(coeffs_int, fs_0)

    # load default coefficients & decimation factors
    # fs_0 = 3072000//4
    decimations = [16, 3]

    # stage 1 parameters
    ma_stages = 6

    # stage 2 parameters
    cutoff = 7000
    transition_bandwidth = 1000
    taps_2 = 48*2
    fir_window = ("kaiser", 6)
    stage_2 = stage_params(cutoff, transition_bandwidth, taps_2, fir_window)

    coeffs = design_2_stage(fs_0, decimations, ma_stages, stage_2, int_coeffs=False)
    coeffs_int = design_2_stage(fs_0, decimations, ma_stages, stage_2, int_coeffs=True)
    coeffs_int = [ft.normalise_coeffs(np.float64(coeffs_int[0][0])), coeffs_int[0][1]], [ft.normalise_coeffs(np.float64(coeffs_int[1][0])), coeffs_int[1][1]]

    fs_1_2 = fs_0 / decimations[0]
    fs_2_2 = fs_1_2 / decimations[1]


    pc.plot_stage(coeffs_int[0], fs_0, fs_2_2, axs=ax0)
    pc.plot_stage(coeffs_int[1], fs_1_2, fs_2_2, axs=ax1)
    pc.plot_filters(coeffs_int, fs_0, axs=ax2)

    # get PDM files
    # pdm_path = Path(r"C:\Users\allanskellett\OneDrive - Xmos\Docs\029_mic_testing\029_mic_testing\768_pdm_xscope.npy")
    # pdm_signal = np.load(pdm_path, allow_pickle=True)

    out_signal_2 = process_pdm(pdm_signal, coeffs[0][0], coeffs[1][0], decimations[0], decimations[1])
    out_signal_int_2 = process_pdm(pdm_signal, coeffs_int[0][0], coeffs_int[1][0], decimations[0], decimations[1])




    plt.figure()
    nfft = 128*2*8
    plt.psd(pdm_signal, Fs=fs_0, NFFT=nfft*(decimations[0]*decimations[1]), color='g', label='PDM')
    plt.psd(out_signal_int, Fs=fs_2, NFFT=int(nfft), label='32-1.5')
    plt.psd(out_signal_int_2, Fs=fs_2_2 , NFFT=nfft, label='16-3')
    # plt.psd(out_signal_int - out_signal_int_2, Fs=fs_2, NFFT=nfft, label='Filtered int coeffs')

    plt.legend()
    plt.title("768 PDM conversion, %s, during sweep" % pdm_path.name)
    plt.xlim([0, fs_0/2])
    plt.xlim([0, 10000])
    plt.yticks(np.arange(-200, 0, 5))
    ax = plt.gca()
    ax.set_xscale('log')
    # plt.xlim([10, fs_0/2])
    plt.xlim([20, 10000])
    # plt.ylim([-130, -70])
    plt.ylim([-140, -90])

    # plt.figure()
    # plt.title("Int error between float and int coeffs")
    # # plt.psd(out_signal - out_signal_int, Fs=fs_2, NFFT=nfft)
    # plt.psd(out_signal_int - out_signal_int_2, Fs=fs_2, NFFT=nfft, label='Filtered int coeffs')

    # plt.xlim([10, 6000])

    plt.figure()
    plt.plot(20*np.log10(np.abs(out_signal_int)), label='32-1.5')
    # plt.plot(20*np.log10(np.abs(out_signal_int_2)), label='16-3')
    plt.grid(True)
    # plt.xlim([0, 9000/32/1.5])



    axs[3].plot(np.arange(len(out_signal_int))*32*1.5 - 1198, 20*np.log10(np.abs(out_signal_int)), label='32-1.5')
    # axs[3].plot(np.arange(len(out_signal_int))*32*1.5 - 1198, (out_signal_int), label='32-1.5')
    axs[3].set_title("decimated signal")
    axs[3].grid(True)
    axs[3].set_xlim([0, 15000])
    # axs[3].set_ylim([-95, 5])
    axs[3].set_ylabel("dB")
    axs[3].set_xlabel("Time (PDM samples)")


    plt.show()
    pass