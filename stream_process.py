import uhd
import numpy as np
import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import time 
  
NUM_RECV_FRAMES = 2040 
BUFFER_STRIDE = 50 
 
def run_sdr(q, quit, update_params, rate, center_freq, gain, device="lime",):
    if device == "uhd": 
        usrp = uhd.usrp.MultiUSRP() 
        usrp.set_rx_rate(rate, 0) 
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq, 0))  
        usrp.set_rx_gain(gain, 0)
 
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
 
        sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
        sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)  
        #sdr.setGain(SOAPY_SDR_RX, 0, gain)

        rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        sdr.activateStream(rxStream)
    # buffer_samps = streamer.get_max_num_samps()
    # print(buffer_samps)
    recv_buffer = np.zeros(NUM_RECV_FRAMES, dtype=np.complex64)
    samples = np.zeros(NUM_RECV_FRAMES * BUFFER_STRIDE, dtype=np.complex64)

    QUEUE_FULL = 0
    QUEUE_WRITTEN = 0

    
    while quit.is_set() is False:
        if not update_params.empty():
            param = update_params.get()
            if param[0] == "freq":
                if device == "uhd":
                    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(float(param[1])), 0)
                    print("set freq")
                else:
                    sdr.setFrequency(SOAPY_SDR_RX, 0, float(param[1]))
            elif param[0] == "gain":
                if device == "uhd":
                    usrp.set_rx_gain(param[1], 0)
                else:
                    sdr.setGain(SOAPY_SDR_RX, 0, param[1])
        else:
            for i in range(BUFFER_STRIDE):
                if device == "uhd":
                    streamer.recv(recv_buffer, metadata)
                else:
                    sr = sdr.readStream(rxStream, [recv_buffer], len(recv_buffer))
                samples[i * NUM_RECV_FRAMES : (i + 1) * NUM_RECV_FRAMES] = recv_buffer
            try:
                q.put_nowait(samples)
                QUEUE_WRITTEN += 1
            except:
                QUEUE_FULL += 1

    print("Queue was full: ", QUEUE_FULL)
    print("Queue was written: ", QUEUE_WRITTEN)
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)
    print("Cleaned usrp")
