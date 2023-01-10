import matplotlib.pyplot as plt
import numpy as np
import os

# # file-based method
# dir_to_inspect = "checkpoints/hulkL_aug_crop_2"
# files = os.listdir(dir_to_inspect)
# files.sort()

# losses = []
# for file in files:
#     if file.endswith(".pth"):
#         losses.append(float(file.split("_")[-1][:-4]))

# plt.plot(losses)
# plt.savefig("losses.png")

# npy file-based method
files_to_inspect = ['/home/vainavi/hulk-keypoints/checkpoints/2023-01-09-23-31-12_UNDER_OVER_NONE/test_losses_2023-01-09-23-31-12_UNDER_OVER_NONE.npy', 
                    '/home/vainavi/hulk-keypoints/checkpoints/2023-01-09-23-31-12_UNDER_OVER_NONE/train_losses_2023-01-09-23-31-12_UNDER_OVER_NONE.npy']
losses = [np.load(file_to_inspect) for file_to_inspect in files_to_inspect]

# smooth losses
# losses = np.convolve(losses, np.ones((80,))/80, mode='valid')
for i, loss in enumerate(losses):
    plt.plot(loss, label=files_to_inspect[i])
plt.legend()
plt.savefig(f"losses_smoothed_compared.png")
