import numpy as np

file=open("cos_wave.iq", "rb")
data = file.read(8 * 50)
dt = np.dtype(np.complex64)
new = np.frombuffer(data, dtype=np.complex64)
print(new)
