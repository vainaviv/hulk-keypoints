import matplotlib.pyplot as plt
import os

dir_to_inspect = "checkpoints/loop_detectortrain_test_loss"
files = os.listdir(dir_to_inspect)
files.sort()

train_losses = []
test_losses = []
for file in files:
    if file.endswith(".pth"):
        test_losses.append(float(file.split("_")[-1][:-4]))
        train_losses.append(float(file.split("_")[-2]))

print(len(train_losses))
print(len(test_losses))

plt.plot(test_losses)
plt.savefig("test_loss.png")

plt.plot(train_losses)
plt.savefig("train_loss.png")