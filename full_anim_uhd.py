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
NUM_RECV_FRAMES = 400

usrp.set_rx_rate(SAMPLE_RATE, 0)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(CENTER_FREQ), 0)
usrp.set_rx_gain(GAIN, 0)

st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
metadata = uhd.types.RXMetadata()
streamer = usrp.get_rx_stream(st_args)
recv_buffer = np.zeros((1, NUM_RECV_FRAMES), dtype=np.complex64)

stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
streamer.issue_stream_cmd(stream_cmd)

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'r')

def init():
    xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)
    ax.set_xlim(min(xf), max(xf))
    ax.set_ylim(-1, 3)
    return ln,

def update(frame):
    xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)
    
    samples = np.zeros(NUM_SAMPS, dtype=np.complex64)
    for i in range(NUM_SAMPS//NUM_RECV_FRAMES):
        streamer.recv(recv_buffer, metadata)
        samples[i * NUM_RECV_FRAMES:(i + 1) * NUM_RECV_FRAMES] = recv_buffer[0]
    window = signal.windows.hann(NUM_SAMPS)
    yf = fft(samples * window)
    yf = fftshift(yf)
    yf = np.abs(yf)
    std = np.std(yf)
    yf = np.clip(yf, std / 3, std * 3)

    ln.set_data(xf, yf)
    return ln,

ani = FuncAnimation(fig, update, frames=1000, init_func=init, blit=True)
plt.show()

stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
streamer.issue_stream_cmd(stream_cmd)
