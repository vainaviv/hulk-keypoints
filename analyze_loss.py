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
files_to_inspect = ['/home/vainavi/hulk-keypoints/checkpoints/2023-01-13-23-02-52_TRCR28_CL3_8_PL1_MED3_UNet50_B64_OS_RotCond_Medley_MoreReal_Sharp/test_losses.npy', 
                    '/home/vainavi/hulk-keypoints/checkpoints/2023-01-13-23-02-52_TRCR28_CL3_8_PL1_MED3_UNet50_B64_OS_RotCond_Medley_MoreReal_Sharp/train_losses.npy']
losses = [np.load(file_to_inspect) for file_to_inspect in files_to_inspect]

# smooth losses
# losses = np.convolve(losses, np.ones((80,))/80, mode='valid')
for i, loss in enumerate(losses):
    plt.plot(loss, label=files_to_inspect[i].split('/')[-1])
plt.legend()
plt.savefig(f"losses_smoothed_compared.png")
