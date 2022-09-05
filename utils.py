from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import butter, lfilter
from numpy import abs as np_abs
def get_fft(samples):
    fft_result = fft(samples)
    fft_result = np_abs(fftshift(fft_result))
    return fft_result

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
