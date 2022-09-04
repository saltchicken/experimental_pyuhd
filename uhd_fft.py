import matplotlib.pyplot as plt
# plt.style.use("ggplot")
plt.style.use("dark_background")
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, Slider
from scipy.signal import windows, find_peaks, peak_widths
import time, multiprocessing
from multiprocessing.sharedctypes import Value
from ctypes import c_double

from stream_process import run_sdr
from utils import get_fft, fftshift, fftfreq

import argparse
import code

NUM_SAMPS = 1600
MAX_QUEUE_SIZE = 50

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", default="", type=str)
	# parser.add_argument("-o", "--output-file", type=str, required=True)
    parser.add_argument("-f", "--freq", default=104500000, type=float)
    parser.add_argument("-r", "--rate", default=20e6, type=float)
    # parser.add_argument("-d", "--duration", default=5.0, type=float)
    # parser.add_argument("-c", "--channels", default=0, nargs="+", type=int)
    parser.add_argument("-g", "--gain", default=0, type=int)
    return parser.parse_args()

def fft_process(q, quit):
    window = windows.hann(NUM_SAMPS)
    while quit.is_set() is False:
        try:
            data = q.get()
            data = data.astype("complex64")
            data.resize(data.size//NUM_SAMPS, NUM_SAMPS)
            for i in range(data.size//NUM_SAMPS):
                data[i * NUM_SAMPS: (i + 1) * NUM_SAMPS] = get_fft(data[i * NUM_SAMPS: (i + 1) * NUM_SAMPS] * window)
            data = data.mean(axis=0)
            try:
                output_q.put_nowait(data)
            except:
                pass
        except:
            pass
    print("FFT closed")



def matplotlib_process(out_q, quit, update_params, rate, center_freq, gain_init):
    class Index:
        def __init__(self, quit, update_params):
            self.quit = quit
            self.update_params = update_params
            self.center_freq = center_freq
        def start(self, event):
            # Not implemented
            pass
        def stop(self, event):
            self.quit.set()
        def change_freq(self, freq):
            if freq != '':
                self.center_freq.value = float(freq)
                self.update_params.put(("freq", self.center_freq.value))
        def change_gain(self, gain):
            self.update_params.put(("gain", gain))
        def on_press(self, event):
            if event.key == "right":
                self.center_freq.value += 100000.0
                self.update_params.put(("freq", self.center_freq.value))
            elif event.key == "left":
                self.center_freq.value -= 100000.0
                self.update_params.put(("freq", self.center_freq.value))
            else: print(event.key)

    def update(frame, ax, xf, fft_line, peak_graph, center_freq):
        try:
            while not output_q.empty(): 
                data = output_q.get()
            data = data.astype("complex64")
            # yf = get_fft(data[-NUM_SAMPS:] * window)
            # print("FFT analysis took {:.4f} seconds".format(toc-tic))
            peaks, _ = find_peaks(data, height=1)
            # results_half = peak_widths(data, peaks, rel_height=0.5)

            # TODO Move this out of loop, this causes the lag when holding arrow key
            xf = fftshift(fftfreq(NUM_SAMPS, 1 / rate) + float(center_freq.value))
            ax.set_xlim(min(xf), max(xf))
            ax.set_ylim(-1, 6)

            fft_line.set_data(xf, data)
            peak_graph.set_data(xf[peaks], data[peaks])
            return fft_line, peak_graph
        except:
            #print("error with update in plot_processing")
            return fft_line, peak_graph

    output_q = out_q
    fig, ax = plt.subplots(figsize=(12, 10))
    fft_line = plt.plot([], [], 'r')[0]
    peak_graph = plt.plot([], [], 'x')[0]

    callback = Index(quit, update_params)
    axstop = plt.axes([0.7, 0.0, 0.075, 0.02])
    bstop = Button(axstop, 'Stop')
    bstop.on_clicked(callback.stop)

    axtext = plt.axes([0.1, 0.05, 0.3, 0.02])
    text_box = TextBox(axtext, "Center_Freq", textalignment="center")
    text_box.on_submit(callback.change_freq)

    axgain = plt.axes([0.1, 0.25, 0.0225, 0.63])
    gain_slider = Slider(ax=axgain, label="Gain", valmin=0, valmax=50, valinit=gain_init, orientation="vertical")
    gain_slider.on_changed(callback.change_gain)


    fig.canvas.mpl_connect('key_press_event', callback.on_press)

    xf = fftshift(fftfreq(NUM_SAMPS, 1 / rate) + float(center_freq.value))

    ax.set_xlim(min(xf), max(xf))
    ax.set_ylim(-1, 6)

    ani = FuncAnimation(fig, update, frames=None, fargs=(ax, xf, fft_line, peak_graph, center_freq), interval=0, blit=False)
    plt.show()
    print("Setting quit")
    quit.set()

if __name__ == "__main__":

    args = parse_args()

    q = multiprocessing.Queue( MAX_QUEUE_SIZE )
    quit = multiprocessing.Event()
    update_params = multiprocessing.Queue(1)
    output_q = multiprocessing.Queue(10)

    center_freq = Value(c_double, args.freq)
    
    run_matplotlib_process=multiprocessing.Process(None, matplotlib_process, args=(output_q, quit, update_params, args.rate, center_freq, args.gain))
    run_matplotlib_process.start()

    run_FFT_process=multiprocessing.Process(None, fft_process, args=(q, quit))
    run_FFT_process.start()
    run_FFT_process2=multiprocessing.Process(None, fft_process, args=(q, quit))
    run_FFT_process2.start()

    run_sdr_process=multiprocessing.Process(None, run_sdr, args=(q, quit, update_params, args.rate, center_freq, args.gain, "uhd"))
    run_sdr_process.start()

    while quit.is_set() is False:
        time.sleep(1)
    print("plot closed")
    quit.set()
    time.sleep(1)
    run_matplotlib_process.terminate()
    run_sdr_process.terminate()
    run_FFT_process.terminate()
    run_FFT_process2.terminate()
    run_matplotlib_process.join()
    run_sdr_process.join()
    run_FFT_process.join()
    run_FFT_process2.join()
    q.close()
    output_q.close()
    update_params.close()
    print("Cleaned everything")
