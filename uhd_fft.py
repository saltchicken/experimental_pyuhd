import matplotlib.pyplot as plt
plt.style.use("ggplot")
# plt.style.use("dark_background")
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, Slider, RadioButtons
from scipy.signal import windows, find_peaks, peak_widths
import time, multiprocessing
from multiprocessing.sharedctypes import Value
from ctypes import c_double

from stream_process import run_sdr
from utils import get_fft, set_xf, butter_lowpass_filter, SignalGen

import numpy as np
import argparse

import datetime as dt
import sigmf
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str

DOWNSAMPLE = 4

def saveSigMF(data, sample_rate, freq, title, description=None):
    # data.tofile('{}.sigmf-data'.format(title))
    
    meta = SIgMFFile(
            data_file='{}.sigmf-data'.format(title),
            global_info = {
            SigMFFile.DATATYPE_KEY : get_data_type_str(data),
            SigMFFile.SAMPLE_RATE_KEY: int(sample_rate),
            SigMFFile.AUTHOR_KEY: "ANONYMOUS",
            SigMFFile.DESCRIPTION_KEY: description,
            SigMFFile.VERSION_KEY: sigmf.__version__,
            }
        )
    meta.add_capture(0, metadata={
        SigMFFile.FREQUENCY_KEY: freq,
        SigMFFile.DATETIME_KEY: title,
        # SigMFFile.DATETIME_KEY: dt.datetime.utcnow().isoformat()+'Z',
        })
    assert meta.validate()
    meta.tofile('{}.sigmf-meta'.format(title))

def fft_process(sdr_queue, fft_queue, quit, update_offset_freq, offset_freq, fft_size):
    window = windows.hann(fft_size)
    sig = SignalGen(700000, 20000000 // DOWNSAMPLE)
    # cos_file = open("cos_wave.iq", "rb")
    # cos_data = cos_file.read(8 * 24000)
    # cos_wave = np.frombuffer(cos_data, dtype=np.complex64)
    title = dt.datetime.utcnow().isoformat()+'Z'
    with open("{}.bin".format(title), "wb") as bin_file:
#    pool = multiprocessing.Pool(processes=2)
        while quit.is_set() is False:
            if update_offset_freq.is_set():
                # print(offset_freq.value)
                sig = SignalGen(offset_freq.value, 20000000 // DOWNSAMPLE)
                update_offset_freq.clear()
            try:
                data = sdr_queue.get()
            except:
                continue
            data = data.astype("complex64")
            # Low Pass Failter
            # data = butter_lowpass_filter(data, 300000, 20000000, 6)
            data = data[::DOWNSAMPLE]
            extra_samps = data.size % fft_size
            if extra_samps:
                data = data[:-extra_samps]
            mod_sig = sig.slice(data.size)
            data = data / mod_sig
            data.tofile(bin_file)

            # result = [pool.apply(parse_data, args=(i, data, fft_size, window)) for i in range(data.size//fft_size)]

            data = data.reshape(data.size//fft_size, fft_size)
            fft_data = np.zeros((data.shape[0], data.shape[1]))
            for i, batch in enumerate(data):
                fft_data[i] = get_fft(batch * window)
            fft_data = fft_data.mean(axis=0)
            try:
                fft_queue.put_nowait(fft_data)
            except:
                pass
    print("FFT closed")

def matplotlib_process(fft_queue, quit, update_params, update_offset_freq, rate, center_freq, offset_freq, gain, fft_size):
    class Index:
        def __init__(self, ax, quit, update_params, update_offset_freq):
            self.ax = ax
            self.quit = quit
            self.update_params = update_params
            self.update_offset_freq = update_offset_freq
            self.center_freq = center_freq
            self.offset_freq = offset_freq
            self.gain = gain
            self.threshold = 1.0
            self.threshold_line = ax.axhline(self.threshold, 0, 1)
            self.threshold_line.set_visible(False)
            self.xf = set_xf(self.ax, fft_size, rate.value // DOWNSAMPLE, center_freq.value)
            
            self.ax.set_ylim(-1, 6)

        # def start(self, event):
        #     # Not implemented
        #     pass
        
        # def stop(self, event):
        #     self.quit.set()
        
        def change_freq(self, freq):
            if freq != '':
                self.center_freq.value = float(freq)
                self.update_params.set()
                self.xf = set_xf(self.ax, fft_size, rate.value, center_freq.value)
        
        def change_gain(self, gain):
            self.gain.value = int(gain)
            self.update_params.set()
        
        def on_press(self, event):
            if event.key == "right":
                self.center_freq.value += 100000.0
                self.update_params.set()
                self.xf = set_xf(self.ax, fft_size, rate.value, center_freq.value)
            elif event.key == "left":
                self.center_freq.value -= 100000.0
                self.update_params.set()
                self.xf = set_xf(self.ax, fft_size, rate.value, center_freq.value)
        
        def threshold_clicked(self, label):
            if label == "On":
                self.threshold_line.set_visible(True)
            else:
                self.threshold_line.set_visible(False)

        def change_threshold(self, threshold):
            self.threshold = threshold
            self.threshold_line.set_ydata([self.threshold, self.threshold])

        def change_offset_freq(self, freq):
            if freq != '':
                self.offset_freq.value = float(freq)
                self.update_offset_freq.set()

        def update(self, frame, fft_line, peak_graph):
            try:
                while not fft_queue.empty(): 
                    data = fft_queue.get()
                data = data.astype("complex64")
                # yf = get_fft(data[-fft_size:] * window)
                # print("FFT analysis took {:.4f} seconds".format(toc-tic))
                # TODO Fix Complex Error warning
                peaks, _ = find_peaks(data, height=self.threshold)
                # results_half = peak_widths(data, peaks, rel_height=0.5)

                fft_line.set_data(self.xf, data)
                peak_graph.set_data(self.xf[peaks], data[peaks])
                return fft_line, peak_graph
            except:
                #print("error with update in plot_processing")
                return fft_line, peak_graph

    fig, ax = plt.subplots(figsize=(12, 10))
    fft_line = plt.plot([], [], 'r')[0]
    peak_graph = plt.plot([], [], 'x')[0]

    callback = Index(ax, quit, update_params, update_offset_freq)

    # axstop = plt.axes([0.7, 0.0, 0.075, 0.02])
    # bstop = Button(axstop, 'Stop')
    # bstop.on_clicked(callback.stop)

    ax_center_freq_textbox = plt.axes([0.1, 0.05, 0.3, 0.02])
    center_freq_textbox = TextBox(ax_center_freq_textbox, "Center_Freq", textalignment="center")
    center_freq_textbox.on_submit(callback.change_freq)

    ax_gain_slider = plt.axes([0.02, 0.25, 0.0225, 0.63])
    gain_slider = Slider(ax=ax_gain_slider, label="Gain", valmin=0, valmax=50, valinit=gain.value, orientation="vertical")
    gain_slider.on_changed(callback.change_gain)
    
    ax_threshold_slider = plt.axes([0.075, 0.25, 0.0225, 0.63])
    threshold_slider = Slider(ax=ax_threshold_slider, label="Threshold", valmin=0, valmax=6, valinit=callback.threshold, orientation="vertical")
    threshold_slider.on_changed(callback.change_threshold)

    ax_threshold_radio = plt.axes([0.07, 0.15, 0.03, 0.05])
    threshold_radio = RadioButtons(ax_threshold_radio, ("On", "Off"), 1)
    threshold_radio.on_clicked(callback.threshold_clicked)

    ax_offset_freq_slider = plt.axes([0.13, 0.25, 0.0225, 0.63])
    offset_freq_slider = Slider(ax=ax_offset_freq_slider, label="Offset", valmin=-10000000, valmax=10000000, valinit=0, valstep=100000, orientation="vertical")
    offset_freq_slider.on_changed(callback.change_offset_freq)

    fig.canvas.mpl_connect('key_press_event', callback.on_press)

    ani = FuncAnimation(fig, callback.update, frames=None, fargs=(fft_line, peak_graph), interval=0, blit=False)
    plt.show()
    print("Setting quit")
    quit.set()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", default="uhd", type=str)
	# parser.add_argument("-o", "--output-file", type=str, required=True)
    parser.add_argument("-f", "--freq", default=105000000, type=float)
    parser.add_argument("-r", "--rate", default=20e6, type=float)
    # parser.add_argument("-d", "--duration", default=5.0, type=float)
    # parser.add_argument("-c", "--channels", default=0, nargs="+", type=int)
    parser.add_argument("-g", "--gain", default=0, type=int)
    parser.add_argument("--fft_size", default=2000, type=int)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    sdr_queue = multiprocessing.Queue(8)
    fft_queue = multiprocessing.Queue(8)
    quit = multiprocessing.Event()
    update_params = multiprocessing.Event()
    update_offset_freq = multiprocessing.Event()

    center_freq = Value(c_double, args.freq)
    offset_freq = Value(c_double, 0)
    gain = Value('i', args.gain)
    rate = Value(c_double, args.rate)
    fft_size = args.fft_size
    
    run_matplotlib_process=multiprocessing.Process(None, matplotlib_process, args=(fft_queue, quit, update_params, update_offset_freq, rate, center_freq, offset_freq, gain, fft_size))
    run_matplotlib_process.start()
    
    fft_process_list = [multiprocessing.Process(None, fft_process, args=(sdr_queue, fft_queue, quit, update_offset_freq, offset_freq, fft_size)) for _ in range(1)]

    for proc in fft_process_list:
        proc.start()

    run_sdr_process=multiprocessing.Process(None, run_sdr, args=(sdr_queue, quit, update_params, rate, center_freq, gain, args.args))
    run_sdr_process.start()

    while quit.is_set() is False:
        time.sleep(1)
    print("plot closed")
    quit.set()
    time.sleep(1)
    run_matplotlib_process.terminate()
    run_sdr_process.terminate()
    for proc in fft_process_list:
        proc.terminate()
    run_matplotlib_process.join()
    run_sdr_process.join()
    for proc in fft_process_list:
        proc.join()
    sdr_queue.close()
    fft_queue.close()
    print("Cleaned everything")
