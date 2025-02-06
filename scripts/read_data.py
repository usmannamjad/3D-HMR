import numpy as np


path = "/home/usmannamjad/vh/tram/results/example_video/hps/hps_track_0.npy"

data = np.load(path, allow_pickle=True)

print(type(data))
print(data.shape)