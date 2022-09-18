from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
import math

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
    def __init__(self, freq, fs):
        self.fs = fs
        self.freq = freq
        self.index = 0
        if freq:
            self.period = self.fs / self.freq
        else: self.period = 1
        if self.period != int(self.period):
            print("WARNING: Frequency is not perfect divisor of sample rate")
    def slice(self, size):
        phi = 0.0
        wt = np.array(2 * np.pi * self.freq * np.arange(self.index, self.index + size) / self.fs, dtype=np.complex64)
        sig_complex = np.exp(1j * (wt + phi))
        # result = np.cos(x * np.pi * 2 * self.freq) + 1j*np.sin(x * np.pi * 2 * self.freq)
        self.index += size
        if self.index >= self.period:
            self.index = self.index % math.ceil(self.period)
        return sig_complex

def show_signal():
    fig, ax = plt.subplots(figsize=(6,6))
    fs = 20e6
    
    # bb_freq = fs / 8  # baseband frequency of tone
    bb_freq = 700000  # baseband frequency of tone
    fcen = bb_freq

    period = fs / fcen

    # phi = 0.285
    phi = 0.0
    buff_len = int(math.ceil(period) )
    # print("buff_len ", buff_len, "period ", period)
    # assert buff_len % period == 0, 'Total samples not integer number of periods'
    wt = np.array(2 * np.pi * fcen * np.arange(buff_len) / fs)
    sig_complex = np.exp(1j * (wt + phi))
    # print(sig_complex[0], sig_complex[-1])
    sig_int16 = np.empty(2 * buff_len, dtype=np.int16)
    sig_int16[0::2] = 32767 / 4 * sig_complex.real
    sig_int16[1::2] = 32767 / 4 * sig_complex.imag
    # freq = 106e6  
    # tx_buff = make_tone(buff_len, bb_freq, fs)
    # print(len(tx_buff))  
    # print(tx_buff.shape)  
    # tx_buff2 = make_tone(buff_len, bb_freq / 2, fs)
    # tx_buff = tx_buff + tx_buff2  
      
    
    sig = SignalGen(20, 2000)
    sig_slice = sig.slice(75)
    test = np.concatenate((sig_slice, sig.slice(75)))
    # lo_freq = freq - bb_freq  # Calc LO freq to put tone at tnp.arange(buff_len)/fs
    ax.plot( test.real )
    ax.plot( test.imag )
    plt.show()


if __name__ == "__main__":
#     sig = SignalGen(1000, 20e6)
#     sig.slice(1000)
    show_signal()
