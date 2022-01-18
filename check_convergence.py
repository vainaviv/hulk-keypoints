import matplotlib.pyplot as plt
import os
import pickle

dir_to_inspect = "checkpoints/detect_ep"
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

data = None
with open('test_accuracies.pkl', 'rb') as f:
    data = pickle.load(f)

test_acc = []
for i in range(len(data)):
    if str(i) in data:
        test_acc.append(data[str(i)])
    else:
        test_acc.append(test_acc[-1])
plt.plot(test_acc, label="test acc")
#############

train_data = None
with open('train_accuracies.pkl', 'rb') as f:
    train_data = pickle.load(f)

train_acc = []
for i in range(len(train_data)):
    if str(i) in train_data:
        train_acc.append(train_data[str(i)])
    else:
        train_acc.append(train_acc[-1])

plt.plot(train_acc, label="train acc")

plt.legend()
plt.savefig("analyze_training.png")