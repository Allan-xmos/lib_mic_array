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
    y = spsig.lfilter(b, [1], x_pdm)[::decimation]
    # y = spsig.fftconvolve(x_pdm, b)[::decimation]
    return y


cic_dtype = np.int16
@njit()
def cic(x_pdm, decimation, stages):
    comb_buffer = np.zeros(stages, dtype=cic_dtype)
    int_buffer = np.zeros(stages, dtype=cic_dtype)

    N = x_pdm.shape[0] // decimation
    y = np.zeros(N, dtype=cic_dtype)
    int_out = np.zeros(1, dtype=cic_dtype)
    for n in range(N):
        for m in range(decimation):
            idx = n*decimation + m
            int_out[0] = x_pdm[idx] + np.sum(int_buffer)
            int_buffer[1:] = int_buffer[:1]
            int_buffer[0] = int_out[0]
            
        y[n] = int_out[0] - np.sum(comb_buffer)

        comb_buffer[1:]= comb_buffer[:-1]
        comb_buffer[0] = int_buffer[-1]

    z = y.astype(np.float32) / decimation
    return z

# run cic once so that numba can do its thing
x = np.random.randint(low=-1, high=1, size=100, dtype=cic_dtype)
_ = cic(x[:100], 3_027_000, 4)