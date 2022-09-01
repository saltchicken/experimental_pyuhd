import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
from scipy.signal import windows, find_peaks, peak_widths
import time, multiprocessing

from stream_process import run_usrp, SAMPLE_RATE, CENTER_FREQ, MAX_QUEUE_SIZE
from utils import get_fft, fftshift, fftfreq

NUM_SAMPS = 400

class Index:

    def __init__(self, quit, update_params):
        self.quit = quit
        self.update_params = update_params

    def start(self, event):
        # Not implemented
        pass

    def stop(self, event):
        self.quit.set()

    def change_freq(self, expression):
        self.update_params.put(expression)

def init():
    ax.set_xlim(min(xf), max(xf))
    ax.set_ylim(-1, 6)

def update(frame):
    try:
        while not q.empty(): 
            data = q.get()
        data = data.astype("complex64")
        # yf = get_fft(data[-NUM_SAMPS:] * window)

        data.resize(data.size//NUM_SAMPS, NUM_SAMPS)
        #data = data.mean(axis=0)
        for i in range(data.size//NUM_SAMPS):
            data[i * NUM_SAMPS: (i + 1) * NUM_SAMPS] = get_fft(data[i * NUM_SAMPS: (i + 1) * NUM_SAMPS] * window)
        data = data.mean(axis=0)
        peaks, _ = find_peaks(data, height=1)
        # results_half = peak_widths(data, peaks, rel_height=0.5)
        fft_line.set_data(xf, data)
        peak_graph.set_data(xf[peaks], data[peaks])
        return fft_line, peak_graph
    except:
        #print("error with update in plot_processing")
        return fft_line, peak_graph

q = multiprocessing.Queue( MAX_QUEUE_SIZE )
quit = multiprocessing.Event()
update_params = multiprocessing.Queue(1)


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

xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)

window = windows.hann(NUM_SAMPS)
ani = FuncAnimation(fig, update, init_func=init, frames=None, interval=0, blit=False)

run_usrp_process=multiprocessing.Process(None, run_usrp, args=(q, quit, update_params))
run_usrp_process.start()

plt.show()

print("plot closed")
quit.set()
time.sleep(1)
run_usrp_process.terminate()
run_usrp_process.join()
q.close()
print("Cleaned everything")
