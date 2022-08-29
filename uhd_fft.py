import uhd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
import time, multiprocessing

SAMPLE_RATE = 20e6
NUM_SAMPS = 400
CENTER_FREQ = 104.5e6
GAIN = 50
NUM_RECV_FRAMES = 2040

MAX_QUEUE_SIZE = 100 

def plot_processing(q):
    fig, ax = plt.subplots()
    ln, = plt.plot([], [], 'r')

    xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)
    ax.set_xlim(min(xf), max(xf))
    ax.set_ylim(-1, 6)

    def update(frame):
        try:
            while not q.empty(): 
                data = q.get()
                data = data.astype("complex64")
            window = signal.windows.hann(NUM_SAMPS)
            yf = fft(data[-NUM_SAMPS:] * window)
            yf = fftshift(yf)
            yf = np.abs(yf)
            std = np.std(yf)
            yf = np.clip(yf, std / 3, std * 100)
            ln.set_data(xf, yf)
            return ln,
        except:
            #print("error with update in plot_processing")
            return ln,

    ani = FuncAnimation(fig, update, frames=None, interval=0, blit=True)
    plt.show()

q = multiprocessing.Queue( MAX_QUEUE_SIZE )
plot_process=multiprocessing.Process(None, plot_processing, args=(q,))
plot_process.start()

usrp = uhd.usrp.MultiUSRP()
usrp.set_rx_rate(SAMPLE_RATE, 0)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(CENTER_FREQ), 0)
usrp.set_rx_gain(GAIN, 0)

st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
metadata = uhd.types.RXMetadata()
streamer = usrp.get_rx_stream(st_args)
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
streamer.issue_stream_cmd(stream_cmd)
buffer_samps = streamer.get_max_num_samps()
print(buffer_samps)
recv_buffer = np.zeros(NUM_RECV_FRAMES, dtype=np.complex64)

QUEUE_FULL = 0
QUEUE_WRITTEN = 0

while plot_process.is_alive():
    samples = np.zeros(NUM_RECV_FRAMES * 20, dtype=np.complex64)
    for i in range(20):
        streamer.recv(recv_buffer, metadata)
        samples[i * NUM_RECV_FRAMES : (i + 1) * NUM_RECV_FRAMES] = recv_buffer
    
    if q.qsize() >= MAX_QUEUE_SIZE:
        QUEUE_FULL += 1
        continue
    else:
        QUEUE_WRITTEN += 1
        if QUEUE_WRITTEN == 2:
            print(QUEUE_FULL)
            QUEUE_FULL = 0
        q.put_nowait(samples)

print("Queue was full: ", QUEUE_FULL)
print("Queue was written: ", QUEUE_WRITTEN)
plot_process.terminate()
plot_process.join()
q.close()
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
streamer.issue_stream_cmd(stream_cmd)
print("Cleaned")
