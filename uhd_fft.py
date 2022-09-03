import matplotlib.pyplot as plt
plt.style.use("dark_background")
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, Slider
from scipy.signal import windows, find_peaks, peak_widths
import time, multiprocessing

from stream_process import run_usrp, run_soapy, SAMPLE_RATE, CENTER_FREQ, MAX_QUEUE_SIZE
from utils import get_fft, fftshift, fftfreq

NUM_SAMPS = 1600

def fft_process(q, quit):
    window = windows.hann(NUM_SAMPS)
    while quit.is_set() is False:
        try:
            while not q.empty():
                data = q.get()
            data = data.astype("complex64")
            data.resize(data.size//NUM_SAMPS, NUM_SAMPS)
            for i in range(data.size//NUM_SAMPS):
                data[i * NUM_SAMPS: (i + 1) * NUM_SAMPS] = get_fft(data[i * NUM_SAMPS: (i + 1) * NUM_SAMPS] * window)
            data = data.mean(axis=0)
            output_q.put_nowait(data)
        except:
            pass
    print("FFT closed")


class Index:
    def __init__(self, quit, update_params):
        self.quit = quit
        self.update_params = update_params
    def start(self, event):
        # Not implemented
        pass
    def stop(self, event):
        self.quit.set()
    def change_freq(self, freq):
        self.update_params.put(("freq", freq))
    def change_gain(self, gain):
        self.update_params.put(("gain", gain))

def update(frame, ax, xf, fft_line, peak_graph):
    try:
        while not output_q.empty(): 
            data = output_q.get()
        data = data.astype("complex64")
        # yf = get_fft(data[-NUM_SAMPS:] * window)
        # print("FFT analysis took {:.4f} seconds".format(toc-tic))
        peaks, _ = find_peaks(data, height=1)
        # results_half = peak_widths(data, peaks, rel_height=0.5)
        fft_line.set_data(xf, data)
        peak_graph.set_data(xf[peaks], data[peaks])
        return fft_line, peak_graph
    except:
        #print("error with update in plot_processing")
        return fft_line, peak_graph

def matplotlib_process(out_q, quit, update_params):
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
    gain_slider = Slider(ax=axgain, label="Gain", valmin=0, valmax=50, valinit=10, orientation="vertical")
    gain_slider.on_changed(callback.change_gain)

    xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)

    ax.set_xlim(min(xf), max(xf))
    ax.set_ylim(-1, 6)

    ani = FuncAnimation(fig, update, frames=None, fargs=(ax, xf, fft_line, peak_graph), interval=0, blit=False)
    plt.show()
    print("Setting quit")
    quit.set()

if __name__ == "__main__":
    q = multiprocessing.Queue( MAX_QUEUE_SIZE )
    quit = multiprocessing.Event()
    update_params = multiprocessing.Queue(1)
    output_q = multiprocessing.Queue(10)
    
    run_matplotlib_process=multiprocessing.Process(None, matplotlib_process, args=(output_q, quit, update_params))
    run_matplotlib_process.start()

    run_FFT_process=multiprocessing.Process(None, fft_process, args=(q, quit))
    run_FFT_process.start()

    run_usrp_process=multiprocessing.Process(None, run_usrp, args=(q, quit, update_params))
    run_usrp_process.start()

    while quit.is_set() is False:
        time.sleep(0.5)
    print("plot closed")
    quit.set()
    time.sleep(1)
    run_matplotlib_process.terminate()
    run_usrp_process.terminate()
    run_FFT_process.terminate()
    run_matplotlib_process.join()
    run_usrp_process.join()
    run_FFT_process.join()
    q.close()
    output_q.close()
    update_params.close()
    print("Cleaned everything")
