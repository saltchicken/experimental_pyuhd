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
from utils import get_fft, set_xf, butter_lowpass_filter

import argparse

def fft_process(q, quit, fft_size):
    window = windows.hann(fft_size)
    while quit.is_set() is False:
        try:
            data = q.get()
            data = data.astype("complex64")
            # Low Pass Failter
            # data = butter_lowpass_filter(data, 300000, 20000000, 6)
            # Downsample
            # ratio = 2
            # data = data[::ratio]
            data.resize(data.size//fft_size, fft_size)
            for i in range(data.size//fft_size):
                data[i * fft_size: (i + 1) * fft_size] = get_fft(data[i * fft_size: (i + 1) * fft_size] * window)
            data = data.mean(axis=0)
            try:
                output_q.put_nowait(data)
            except:
                pass
        except:
            pass
    print("FFT closed")

def matplotlib_process(out_q, quit, update_params, rate, center_freq, gain, fft_size):
    class Index:
        def __init__(self, ax, quit, update_params):
            self.ax = ax
            self.quit = quit
            self.update_params = update_params
            self.center_freq = center_freq
            self.gain = gain
            self.threshold = 1.0
            self.threshold_line = ax.axhline(self.threshold, 0, 1)
            self.threshold_line.set_visible(False)
            self.xf = set_xf(self.ax, fft_size, rate.value, center_freq.value)
            
            self.ax.set_ylim(-1, 6)

        # def start(self, event):
        #     # Not implemented
        #     pass
        
        # def stop(self, event):
        #     self.quit.set()
        
        def change_freq(self, freq):
            if freq != '':
                self.center_freq.value = float(freq)
                self.update_params.put(("freq", self.center_freq.value))
                self.xf = set_xf(self.ax, fft_size, rate.value, center_freq.value)
        
        def change_gain(self, gain):
            self.gain.value = int(gain)
            self.update_params.put(("gain", self.gain.value))
        
        def on_press(self, event):
            if event.key == "right":
                self.center_freq.value += 100000.0
                self.update_params.put(("freq", self.center_freq.value))
                self.xf = set_xf(self.ax, fft_size, rate.value, center_freq.value)
            elif event.key == "left":
                self.center_freq.value -= 100000.0
                self.update_params.put(("freq", self.center_freq.value))
                self.xf = set_xf(self.ax, fft_size, rate.value, center_freq.value)
        
        def threshold_clicked(self, label):
            if label == "On":
                self.threshold_line.set_visible(True)
            else:
                self.threshold_line.set_visible(False)

        def change_threshold(self, threshold):
            self.threshold = threshold
            self.threshold_line.set_ydata([self.threshold, self.threshold])

        def update(self, frame, fft_line, peak_graph):
            try:
                while not output_q.empty(): 
                    data = output_q.get()
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

    output_q = out_q
    fig, ax = plt.subplots(figsize=(12, 10))
    fft_line = plt.plot([], [], 'r')[0]
    peak_graph = plt.plot([], [], 'x')[0]

    callback = Index(ax, quit, update_params)

    # axstop = plt.axes([0.7, 0.0, 0.075, 0.02])
    # bstop = Button(axstop, 'Stop')
    # bstop.on_clicked(callback.stop)

    axtext = plt.axes([0.1, 0.05, 0.3, 0.02])
    text_box = TextBox(axtext, "Center_Freq", textalignment="center")
    text_box.on_submit(callback.change_freq)

    ax_gain_slider = plt.axes([0.02, 0.25, 0.0225, 0.63])
    gain_slider = Slider(ax=ax_gain_slider, label="Gain", valmin=0, valmax=50, valinit=gain.value, orientation="vertical")
    gain_slider.on_changed(callback.change_gain)
    
    ax_threshold_slider = plt.axes([0.075, 0.25, 0.0225, 0.63])
    threshold_slider = Slider(ax=ax_threshold_slider, label="Threshold", valmin=0, valmax=6, valinit=callback.threshold, orientation="vertical")
    threshold_slider.on_changed(callback.change_threshold)

    ax_threshold_radio = plt.axes([0.07, 0.15, 0.03, 0.05])
    threshold_radio = RadioButtons(ax_threshold_radio, ("On", "Off"), 1)
    threshold_radio.on_clicked(callback.threshold_clicked)

    fig.canvas.mpl_connect('key_press_event', callback.on_press)

    ani = FuncAnimation(fig, callback.update, frames=None, fargs=(fft_line, peak_graph), interval=0, blit=False)
    plt.show()
    print("Setting quit")
    quit.set()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", default="", type=str)
	# parser.add_argument("-o", "--output-file", type=str, required=True)
    parser.add_argument("-f", "--freq", default=104900000, type=float)
    parser.add_argument("-r", "--rate", default=20e6, type=float)
    # parser.add_argument("-d", "--duration", default=5.0, type=float)
    # parser.add_argument("-c", "--channels", default=0, nargs="+", type=int)
    parser.add_argument("-g", "--gain", default=0, type=int)
    parser.add_argument("--fft_size", default=1600, type=int)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    q = multiprocessing.Queue(8)
    quit = multiprocessing.Event()
    update_params = multiprocessing.Queue(1)
    output_q = multiprocessing.Queue(8)

    center_freq = Value(c_double, args.freq)
    gain = Value('i', args.gain)
    rate = Value(c_double, args.rate)
    fft_size = args.fft_size
    
    run_matplotlib_process=multiprocessing.Process(None, matplotlib_process, args=(output_q, quit, update_params, rate, center_freq, gain, fft_size))
    run_matplotlib_process.start()
    
    fft_process_list = []
    for _ in range(4):
        fft_process_list.append(multiprocessing.Process(None, fft_process, args=(q, quit, fft_size)))

    for proc in fft_process_list:
        proc.start()

    run_sdr_process=multiprocessing.Process(None, run_sdr, args=(q, quit, update_params, rate, center_freq, gain, "uhd"))
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
    q.close()
    output_q.close()
    update_params.close()
    print("Cleaned everything")
