"""
Transmits a tone out of the AIR-T. The script will create a tone segment that
is infinity repeatable without a phase discontinuity and with 8 samples per
period. The TX LO of the AIR-T is set such that the baseband frequency of the
generated tone plus the LO frequency will transmit at the desired RF.
"""
import sys
import numpy as np
import argparse
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_CS16, errToStr
from scipy.signal import butter, lfilter, resample_poly
from scipy.io.wavfile import read

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def make_tone(n, fcen, fs, phi=0.285):
    """
    Generates tone signal window with a frequency that is an integer
    multiple of the sample rate so it can be repeated without a phase
    discontinuity.
    """
    period = fs / fcen
    assert n % period == 0, 'Total samples not integer number of periods'
    # Make Complex Valued Tone Signal
    wt = np.array(2 * np.pi * fcen * np.arange(n) / fs)
    sig_cplx = np.exp(1j * (wt + phi))
    # Convert to interleaved int16 values
    sig_int16 = np.empty(2 * n, dtype=np.int16)
    sig_int16[0::2] = 32767 * sig_cplx.real
    sig_int16[1::2] = 32767 * sig_cplx.imag
    return sig_int16

class SignalGen():
    def __init__(self, fs):
        self.fs = fs
        self.index = 0
        self.step = 1.0 / fs
        self.amp = 1.0
        # self.audio_samp, data = read("voice.wav")
        # self.audio = np.array(data, dtype=float)
        # print(max(self.audio[:, 0]))
        # self.audio = self.audio / 20000 - 0.5
        # print(self.audio)
        # self.audio = resample_poly(self.audio, 20, 1)
    def slice(self, freq, size):
        beg_i = self.index * self.step
        end_i = self.index * self.step + size * self.step
        # print(self.audio[:, 0][self.index:self.index + size].size)
        x = np.linspace(beg_i, end_i, int((end_i - beg_i) / self.step))
        # result = np.cos(self.audio[:, 0][self.index:self.index + size]) + 1j*np.sin(self.audio[:, 0][self.index:self.index + size])
        result = np.cos(x * np.pi * 2 * freq * 1) + 1j*np.sin(x * np.pi * 2 * freq * 1)
        # result = np.cos(x * np.pi * 2 * freq) + 1j*np.zeros(len(x))
        # result = np.zeros(len(x)) + 1j*np.zeros(len(x))
        # result = np.ones(len(x)) + 1j*np.zeros(len(x))
        # result = np.ones(len(x)) + 1j*np.ones(len(x))
        # result = np.zeros(len(x)) + 1j*np.ones(len(x))
        result *= self.amp
        # result = butter_lowpass_filter(result, 100, self.fs, 6)

        # if self.index % 2000000 < 1000000:
        #     # result = np.zeros(len(x)) + 1j*np.zeros(len(x))
        #     result = np.cos(x * np.pi * 2 * freq * 10) + 1j*np.sin(x * np.pi * 2 * freq * 10)


        # print(self.audio[:, 0].size)
        # print(self.audio[:, 1].size)
        # print(result)

        self.index += size
        # TODO why is self.audio.size double the array
        # print(self.audio.size)
        # if self.index >= self.audio[:, 0].size - 16384:
        #     self.index = 0
        # print("Index: " + str(self.index))
        return result


def transmit_tone(freq, chan=0, fs=2e6, gain=20, buff_len=16384, sig_freq=0):
    sig = SignalGen(fs)

    # bb_freq = fs / 8  # baseband frequency of tone
    # tx_buff = make_tone(buff_len, bb_freq, fs)
    # lo_freq = freq - bb_freq  # Calc LO freq to put tone at tone_rf

    # Setup Radio
    args = dict(driver="hackrf")
    sdr = SoapySDR.Device(args)  # Create AIR-T instance
    sdr.setSampleRate(SOAPY_SDR_TX, chan, fs)  # Set sample rate
    sdr.setFrequency(SOAPY_SDR_TX, chan, freq)  # Tune the LO
    sdr.setGain(SOAPY_SDR_TX, chan, gain)

    tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16, [chan])
    sdr.activateStream(tx_stream)  # this turns the radio on

    # Transmit
    print('Now Transmitting')
    while True:
        try:
            tx_buff = sig.slice(sig_freq, buff_len)
            rc = sdr.writeStream(tx_stream, [tx_buff], buff_len)
            if rc.ret != buff_len:
                print('TX Error {}: {}'.format(rc.ret, errToStr(rc.ret)))
        except KeyboardInterrupt:
            break

    # Stop streaming
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)


def parse_command_line_arguments():
    """ Create command line options for transmit function """
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Transmit a tone on the AIR-T',
                                     formatter_class=help_formatter)
    parser.add_argument('-f', type=float, required=False, dest='freq',
                        default=106e6, help='TX Tone Frequency')
    parser.add_argument('-c', type=int, required=False, dest='chan',
                        default=0, help='TX Channel Number [0 or 1]')
    parser.add_argument('-s', type=float, required=False, dest='fs',
                        default=2e6, help='TX Sample Rate')
    parser.add_argument('-g', type=float, required=False, dest='gain',
                        default=20, help='TX gain')
    parser.add_argument('-n', type=int, required=False, dest='buff_len',
                        default=16384, help='TX Buffer Size')
    parser.add_argument('-x', type=int, required=False, dest='x',
                        default=1000, help='Freq of SignalGen')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    pars = parse_command_line_arguments()
    transmit_tone(pars.freq, pars.chan, pars.fs, pars.gain, pars.buff_len, pars.x)
