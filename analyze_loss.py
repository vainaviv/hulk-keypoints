import matplotlib.pyplot as plt
import os

dir_to_inspect = "checkpoints/hulkL_aug_crop_2"
files = os.listdir(dir_to_inspect)
files.sort()

losses = []
for file in files:
    if file.endswith(".pth"):
        losses.append(float(file.split("_")[-1][:-4]))

plt.plot(losses)
plt.savefig("losses.png")