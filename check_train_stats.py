import matplotlib.pyplot as plt
import os
import pickle

dir_to_inspect = "checkpoints/corresponding_segment_r50"
files = os.listdir(dir_to_inspect)
files.sort()

train_losses = []
test_losses = []
for file in files:
    if file.endswith(".pth"):
        test_losses.append(float(file.split("_")[-1][:-4]))
        train_losses.append(float(file.split("_")[-2]))

plt.plot(test_losses, label="test loss")
plt.plot(train_losses, label="train loss")
#############

plt.legend()
plt.savefig("analyze_training.png")