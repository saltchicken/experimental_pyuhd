import uhd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
import time

usrp = uhd.usrp.MultiUSRP()

SAMPLE_RATE = 1e6
NUM_SAMPS = 400
CENTER_FREQ = 104.5e6
GAIN = 50
NUM_RECV_FRAMES = 2000

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
recv_buffer = np.zeros((1, NUM_RECV_FRAMES), dtype=np.complex64)


fig, ax = plt.subplots()
ln, = plt.plot([], [], 'r')

xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)
ax.set_xlim(min(xf), max(xf))
ax.set_ylim(-1, 6)

for t in range(1000):
    #plt.gca().cla()
    #Also need to reset limits
    xf = fftshift(fftfreq(NUM_SAMPS, 1 / SAMPLE_RATE) + CENTER_FREQ)

    samples = np.zeros((1, NUM_SAMPS), dtype=np.complex64)
    # for i in range(NUM_SAMPS//NUM_RECV_FRAMES):
    #     streamer.recv(recv_buffer, metadata)
    #     samples[i * NUM_RECV_FRAMES:(i + 1) * NUM_RECV_FRAMES] = recv_buffer[0]
    

    recv_samps = 0
    while recv_samps < NUM_SAMPS:
        samps = streamer.recv(recv_buffer, metadata)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(metadata.strerror())
        if samps:
            real_samps = min(NUM_SAMPS - recv_samps, samps)
            samples[:, recv_samps:recv_samps + real_samps] = recv_buffer[:, 0:real_samps]
            recv_samps += real_samps
    

    window = signal.windows.hann(NUM_SAMPS)
    yf = fft(samples * window)
    yf = fftshift(yf)
    yf = np.abs(yf)
    std = np.std(yf)
    yf = np.clip(yf, std / 3, std * 100)

    ln.set_data(xf, yf)
    #fig.canvas.draw()
    plt.pause(0.01)

stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
streamer.issue_stream_cmd(stream_cmd)
