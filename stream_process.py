import uhd
import numpy as np
import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import time

SAMPLE_RATE = 20e6
CENTER_FREQ = 104.5e6
GAIN = 10
NUM_RECV_FRAMES = 2040

MAX_QUEUE_SIZE = 50 
BUFFER_STRIDE = 50

def run_usrp(q, quit, update_params):
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
    # print(buffer_samps)
    recv_buffer = np.zeros(NUM_RECV_FRAMES, dtype=np.complex64)
    samples = np.zeros(NUM_RECV_FRAMES * BUFFER_STRIDE, dtype=np.complex64)

    QUEUE_FULL = 0
    QUEUE_WRITTEN = 0

    elapsed = 0.0
    
    while quit.is_set() is False:
        if not update_params.empty():
            param = update_params.get()
            if param[0] == "freq":
                usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(float(param[1])), 0)
            elif param[0] == "gain":
                usrp.set_rx_gain(param[1], 0)
        else:
            tic = time.time()
            for i in range(BUFFER_STRIDE):
                streamer.recv(recv_buffer, metadata)
                samples[i * NUM_RECV_FRAMES : (i + 1) * NUM_RECV_FRAMES] = recv_buffer
            toc = time.time()
            elapsed += toc-tic
            if q.qsize() < MAX_QUEUE_SIZE:
                QUEUE_WRITTEN += 1
                q.put_nowait(samples)
            else:
                QUEUE_FULL += 1
            if elapsed >= 3.0:
                print("{:.2f}".format(QUEUE_WRITTEN / QUEUE_FULL))
                elapsed = 0.0

    print("Queue was full: ", QUEUE_FULL)
    print("Queue was written: ", QUEUE_WRITTEN)
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)
    print("Cleaned usrp")


def run_soapy(q, quit, update_params, device = "hackrf"):
    args = dict(driver="hackrf")
    sdr = SoapySDR.Device(args)

    sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ)
    #sdr.setGain(SOAPY_SDR_RX, 0, GAIN)

    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)

    recv_buffer = np.zeros(NUM_RECV_FRAMES, dtype=np.complex64)
    samples = np.zeros(NUM_RECV_FRAMES * BUFFER_STRIDE, dtype=np.complex64)

    QUEUE_FULL = 0
    QUEUE_WRITTEN = 0

    while quit.is_set() is False:
        if not update_params.empty():
            param = update_params.get()
            if param[0] == "freq":
                sdr.setFrequency(SOAPY_SDR_RX, 0, float(param[1]))
            elif param[0] == "gain":
                sdr.setGain(SOAPY_SDR_RX, 0, param[1])
        else:
            for i in range(BUFFER_STRIDE):
                sr = sdr.readStream(rxStream, [recv_buffer], len(recv_buffer))
                samples[i * NUM_RECV_FRAMES : (i + 1) * NUM_RECV_FRAMES] = recv_buffer

            if q.qsize() < MAX_QUEUE_SIZE:
                QUEUE_WRITTEN += 1
                q.put_nowait(samples)
            else:
                QUEUE_FULL += 1

    print("Queue was full: ", QUEUE_FULL)
    print("Queue was written: ", QUEUE_WRITTEN)
    sdr.deactivateStream(rxStream) #stop streaming
    sdr.closeStream(rxStream)
    print("Cleaned soapy device")
