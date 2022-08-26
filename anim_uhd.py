import uhd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal

usrp = uhd.usrp.MultiUSRP()

SAMPLE_RATE = 20e6
NUM_SAMPS = 400
CENTER_FREQ = 113.0e6
GAIN = 50

fig, ax = plt.subplots()

xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)
# xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)
    
# samples = usrp.recv_num_samps(NUM_SAMPS, CENTER_FREQ, SAMPLE_RATE, [0], GAIN)

# window = signal.windows.hann(NUM_SAMPS)
# yf = fft(samples[0] * window)
# yf = fftshift(yf)
# yf = np.abs(yf)
# std = np.std(yf)
# yf = np.clip(yf, std / 3, std * 3)
ln, = plt.plot([], [], 'r')

def init():
    ax.set_xlim(min(xf), max(xf))
    ax.set_ylim(-1, 3)
    return ln,

def update(frame):
    print(frame)
    xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)
    
    samples = usrp.recv_num_samps(NUM_SAMPS, CENTER_FREQ, SAMPLE_RATE, [0], GAIN)

    window = signal.windows.hann(NUM_SAMPS)
    yf = fft(samples[0] * window)
    yf = fftshift(yf)
    yf = np.abs(yf)
    std = np.std(yf)
    yf = np.clip(yf, std / 3, std * 3)
    ln.set_data(xf, yf)
    return ln,


ani = FuncAnimation(fig, update, frames=1000, init_func=init, blit=True)
plt.show()
