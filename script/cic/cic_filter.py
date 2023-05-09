import numpy as np
import scipy.signal as spsig
from numba import njit

def cic_impulse(decimation, order, differential_delay=1):
    mov_avg_length  = decimation * differential_delay

    # Single stage CIC filter
    # Implemented as generic FIR, because a 'true' CIC filter only works
    # with lossless integer operations.
    h_cic_single = np.ones(mov_avg_length) / mov_avg_length

    # Convolve against itself as much as needed for a higher order CIC filter
    h_cic = h_cic_single
    for i in range(order-1):
        h_cic = np.convolve(h_cic, h_cic_single)

    return h_cic

def cic_ma(x_pdm, decimation, stages):
    b = cic_impulse(decimation, stages)
    y = spsig.resample_poly(x_pdm, 1, decimation, window=b)
    return y

cic_dtype = np.int16
@njit()
def cic(x_pdm, decimation, stages):
    comb_buffer = np.zeros(stages, dtype=cic_dtype)
    tmp = np.zeros(stages, dtype=cic_dtype)
    int_buffer = np.zeros(stages, dtype=cic_dtype)

    N = x_pdm.shape[0] // decimation
    y = np.zeros(N, dtype=cic_dtype)
    for n in range(N):
        for m in range(decimation):
            idx = n*decimation + m
            int_buffer[1:] = int_buffer[1:] + int_buffer[:-1]
            int_buffer[0] = int_buffer[0] + x_pdm[idx]

        tmp[0] = int_buffer[-1] - comb_buffer[0]
        tmp[1:] = tmp[:-1] - comb_buffer[1:]

        y[n] = tmp[-1]

        comb_buffer[1:] = tmp[:-1]
        comb_buffer[0] = int_buffer[-1]
        
    z = y.astype(np.float32) / (decimation ** stages)
    return z

# run cic once so that numba can do its thing
x = np.random.randint(low=-1, high=1, size=100, dtype=cic_dtype)
_ = cic(x[:100], 3_027_000, 4)

def half_band_calc_filter(fs, f_pb, N):
    assert f_pb < fs/4, "A half-band filter requires that Fpb is smaller than Fs/4"
    assert N % 2 == 0, "Filter order N must be a multiple of 2"
    assert N % 4 != 0, "Filter order N must not be a multiple of 4"

    g = spsig.remez(
            N//2+1,
            [0., 2*f_pb/fs, .5, .5],
            [1, 0],
            [1, 1]
            )

    zeros = np.zeros(N//2+1)

    h = [item for sublist in zip(g, zeros) for item in sublist][:-1]
    h[N//2] = 1.0
    h = np.array(h)/2

    return h
