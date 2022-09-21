import uhd
import numpy as np
import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import time 
  
NUM_RECV_FRAMES = 1000 # TODO Why is this working better than 2040 which is the max buff length
BUFFER_STRIDE = 100 # TODO Why does keeping 100000 the product of these two variables the fastest.




def run_sdr(sdr_queue, quit, update_params, rate, center_freq, gain, device):
    if device == "uhd": 
        usrp = uhd.usrp.MultiUSRP() 
        usrp.set_rx_rate(rate.value, 0) 
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq.value, 0))  
        usrp.set_rx_gain(gain.value, 0)
 
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0] 
        metadata = uhd.types.RXMetadata()
        streamer = usrp.get_rx_stream(st_args)
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True 
        streamer.issue_stream_cmd(stream_cmd)
    else: 
        args = dict(driver="lime") 
        sdr = SoapySDR.Device(args) 
 
        sdr.setSampleRate(SOAPY_SDR_RX, 0, rate.value)
        sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq.value)  
        #sdr.setGain(SOAPY_SDR_RX, 0, gain.value)

        rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        sdr.activateStream(rxStream)
    # buffer_samps = streamer.get_max_num_samps()
    # print(buffer_samps)
    recv_buffer = np.zeros(NUM_RECV_FRAMES, dtype=np.complex64)
    samples = np.zeros(NUM_RECV_FRAMES * BUFFER_STRIDE, dtype=np.complex64)

    QUEUE_FULL = 0
    QUEUE_WRITTEN = 0

    
    while quit.is_set() is False:
        if update_params.is_set():
            if device == "uhd":
                usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq.value), 0)
                usrp.set_rx_gain(gain.value, 0)
            else:
                sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq.value)
                sdr.setGain(SOAPY_SDR_RX, 0, gain.value)
            update_params.clear()

        for i in range(BUFFER_STRIDE):
            if device == "uhd":
                streamer.recv(recv_buffer, metadata)
            else:
                sr = sdr.readStream(rxStream, [recv_buffer], len(recv_buffer))
            samples[i * NUM_RECV_FRAMES : (i + 1) * NUM_RECV_FRAMES] = recv_buffer
        try:
            sdr_queue.put_nowait(samples)
            QUEUE_WRITTEN += 1
        except:
            QUEUE_FULL += 1

    print("Queue was full: ", QUEUE_FULL)
    print("Queue was written: ", QUEUE_WRITTEN)
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)
    print("Cleaned usrp")
