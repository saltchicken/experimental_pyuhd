from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
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
    def __init__(self, freq, fs):
        self.fs = fs
        self.index = 0
        self.step = 1.0 / fs
        self.freq = freq
        if freq:
            self.repeat_index = (1 / freq) * fs
        else: self.repeat_index = 1
        if self.repeat_index != int(self.repeat_index):
            print("WARNING: Frequency is not perfect divisor of sample rate")
        self.period = self.fs / self.freq

    def slice(self, size):
        # beg_i = self.index * self.step
        # end_i = self.index * self.step + size * self.step
        # x = np.linspace(beg_i, end_i - self.step, size)
        phi = 0.0
        # t = np.arange(size) / self.fs
        wt = np.array(2 * np.pi * self.freq * np.arange(size) / self.fs, dtype=np.complex64)
        result = np.exp(1j * (wt + phi))
        # result = np.cos(x * np.pi * 2 * self.freq) + 1j*np.sin(x * np.pi * 2 * self.freq)
        self.index += size
        if self.index >= self.repeat_index:
            self.index = self.index % self.repeat_index
        return result

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
    print("buff_len ", buff_len, "period ", period)
    # assert buff_len % period == 0, 'Total samples not integer number of periods'
    wt = np.array(2 * np.pi * fcen * np.arange(buff_len) / fs)
    sig_complex = np.exp(1j * (wt + phi))
    print(sig_complex[0], sig_complex[-1])
    sig_int16 = np.empty(2 * buff_len, dtype=np.int16)
    sig_int16[0::2] = 32767 / 4 * sig_complex.real
    sig_int16[1::2] = 32767 / 4 * sig_complex.imag
    # freq = 106e6  
    # tx_buff = make_tone(buff_len, bb_freq, fs)
    # print(len(tx_buff))  
    # print(tx_buff.shape)  
    # tx_buff2 = make_tone(buff_len, bb_freq / 2, fs)
    # tx_buff = tx_buff + tx_buff2  
      
    
    sig = SignalGen(bb_freq, 20e6)
    sig_slice = sig.slice(1600)
    # lo_freq = freq - bb_freq  # Calc LO freq to put tone at tnp.arange(buff_len)/fs
    ax.plot( sig_slice.real )
    ax.plot( sig_slice.imag )
    plt.show()


# if __name__ == "__main__":
#     sig = SignalGen(1000, 20e6)
#     sig.slice(1000)
    # show_signal()
