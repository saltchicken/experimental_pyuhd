import uhd
import numpy as np

SAMPLE_RATE = 20e6
CENTER_FREQ = 104.5e6
GAIN = 10
NUM_RECV_FRAMES = 2040

MAX_QUEUE_SIZE = 1 

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
    samples = np.zeros(NUM_RECV_FRAMES * 50, dtype=np.complex64)

    QUEUE_FULL = 0
    QUEUE_WRITTEN = 0
    
    while quit.is_set() is False:
        if not update_params.empty():
            param = update_params.get()
            if param[0] == "freq":
                usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(float(param[1])), 0)
            elif param[0] == "gain":
                usrp.set_rx_gain(param[1], 0)
        else:
            for i in range(50):
                streamer.recv(recv_buffer, metadata)
                samples[i * NUM_RECV_FRAMES : (i + 1) * NUM_RECV_FRAMES] = recv_buffer
            
            if q.qsize() < MAX_QUEUE_SIZE:
                QUEUE_WRITTEN += 1
                q.put_nowait(samples)
            else:
                QUEUE_FULL += 1

    print("Queue was full: ", QUEUE_FULL)
    print("Queue was written: ", QUEUE_WRITTEN)
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)
    print("Cleaned usrp")
