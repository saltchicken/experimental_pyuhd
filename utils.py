from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import butter, lfilter
import numpy as np

def get_fft(samples):
    fft_result = fft(samples)
    fft_result = np.abs(fftshift(fft_result))
    return fft_result

def set_xf(ax, fft_size, rate, center_freq):
    xf = fftshift(fftfreq(fft_size, 1 / rate) + float(center_freq))
    ax.set_xlim(min(xf), max(xf))
    return xf

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

class SignalGen():
    def __init__(self, fs):
        self.fs = fs
        self.index = 0
        self.step = 1.0 / fs
    def slice(self, freq, size):
        beg_i = self.index * self.step
        end_i = self.index * self.step + size * self.step
        x = np.linspace(beg_i, end_i, 2000)
        result = np.cos(x * np.pi * 2 * freq) + 1j*np.sin(x * np.pi * 2 * freq)
        self.index += size
        # print(result.size)
        # print(self.index)
        if self.index > 200000:
            self.index = 0
        return result
