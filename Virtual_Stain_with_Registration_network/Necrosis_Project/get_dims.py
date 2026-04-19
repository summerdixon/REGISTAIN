import numpy as np
import glob
import os

bf_images = glob.glob("G:/Project_Nectrotic/Data/ProcessingData/RegistrationRound2_crops/NPY/NonNecrotic/BF/*.npy")

unique_bf_shapes = set()

for i, bf in enumerate(bf_images):
    img = np.load(bf)
    unique_bf_shapes.add(img.shape)
    if i == 100:
        break

print("Unique bf image shapes:" + str(unique_bf_shapes))




af_images = glob.glob("G:/Project_Nectrotic/Data/ProcessingData/RegistrationRound2_crops/NPY/NonNecrotic/AF/*.npy")

unique_af_shapes = set()

for i, af in enumerate(af_images):
    img = np.load(af)
    unique_af_shapes.add(img.shape)
    if i == 100:
        break

print("Unique af image shapes:" + str(unique_af_shapes))


