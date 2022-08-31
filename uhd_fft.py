import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
from scipy.signal import find_peaks, peak_widths
import time, multiprocessing

from uhd_process import *


class Index:
    def __init__(self, quit, update_params):
        self.quit = quit
        self.update_params = update_params
    def start(self, event):
        self.quit.set()
    def stop(self, event):
        self.quit.set()
    def change_freq(self, expression):
        self.update_params.put(expression)

NUM_SAMPS = 400

def update(frame):
    try:
        while not q.empty(): 
            data = q.get()
            data = data.astype("complex64")
        window = signal.windows.hann(NUM_SAMPS)

        # yf = fft(data[-NUM_SAMPS:] * window)
        # yf = fftshift(yf)
        # yf = np.abs(yf)
        # std = np.std(yf)
        # yf = np.clip(yf, std / 30, std * 100)
        

        data.resize(data.size//NUM_SAMPS, NUM_SAMPS)
        #data = data.mean(axis=0)
        for i in range(data.size//NUM_SAMPS):
            yf2 = fft(data[i * NUM_SAMPS: (i + 1) * NUM_SAMPS] * window)
            yf2 = fftshift(yf2)
            yf2 = np.abs(yf2)
            #std = np.std(yf2)
            #yf2 = np.clip(yf2, std / 30, std * 100)
            data[i * NUM_SAMPS: (i + 1) * NUM_SAMPS] = yf2
        data = data.mean(axis=0)
        peaks, _ = find_peaks(data, threshold = 0.5)
        # results_half = peak_widths(data, peaks, rel_height=0.5)
        ln.set_data(xf, data)
        peak_graph.set_data(xf[peaks], data[peaks])
        return ln, peak_graph
    except:
        #print("error with update in plot_processing")
        return ln, peak_graph

q = multiprocessing.Queue( MAX_QUEUE_SIZE )
quit = multiprocessing.Event()
update_params = multiprocessing.Queue(1)


fig, ax = plt.subplots(figsize=(12, 10))
ln = plt.plot([], [], 'r')[0]
peak_graph = plt.plot([], [], 'x')[0]

callback = Index(quit, update_params)
axstop = plt.axes([0.7, 0.0, 0.075, 0.02])
bstop = Button(axstop, 'Stop')
bstop.on_clicked(callback.stop)

axtext = plt.axes([0.1, 0.05, 0.3, 0.02])
text_box = TextBox(axtext, "Center_Freq", textalignment="center")
text_box.on_submit(callback.change_freq)

xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)
ax.set_xlim(min(xf), max(xf))
ax.set_ylim(-1, 6)

ani = FuncAnimation(fig, update, frames=None, interval=0, blit=False)


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
