from scipy.fft import fft, fftfreq, fftshift
from numpy import abs as np_abs
def get_fft(samples):
    fft_result = fft(samples)
    fft_result = np_abs(fftshift(fft_result))
    return fft_result

