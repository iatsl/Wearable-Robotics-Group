import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

temp_dataframe = pd.read_csv("C:\\Users\\Garrett's Laptop\\Desktop\\ICORR_Plot_Revision\\Final Test Results\\StairNet\\StairNet_train_18.csv")
temp_dataframe_2 = pd.read_csv("C:\\Users\\Garrett's Laptop\\Desktop\\ICORR_Plot_Revision\\Final Test Results\\StairNet\\StairNet_val_18.csv")
fig, ax = plt.subplots(figsize=(7, 6))

x = np.arange(0, 61, 10)
y_scale = np.arange(0.82, 1, 0.04)
y = temp_dataframe["accuracy"]
z = temp_dataframe_2["accuracy"]

values = ['1', '10', '20', '30', '40', '50', '60']
y_values = ['0', '0.10', '0.20', '0.30', '0.40', '0.50']

# Set general font size
plt.rcParams['font.size'] = '16'
plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.weight"] = "bold"

# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)
    label.set_fontfamily("Arial")
    # label.set_fontweight("bold")

ax.plot(y, color='maroon', linestyle='dashed', label='Training', linewidth = 3.0)
ax.plot(z, color='midnightblue', label='Validation', linewidth = 3.0)
plt.xlabel('Epochs', fontsize=16, fontfamily = "Arial")
# plt.xlim(0,60)
plt.xticks(x, values)
plt.yticks(y_scale)
plt.ylabel('Accuracy', fontsize=16, fontfamily = "Arial")
plt.subplots_adjust(left=0.2)
plt.subplots_adjust(bottom=0.2)

# fig.suptitle('Sine and cosine waves')
leg = ax.legend()

plt.savefig("C:\\Users\\Garrett's Laptop\\Desktop\\ICORR_Plot_Revision\\Final Test Results\\StairNet\\StairNet_18_Acc_Plot_2.png", dpi=2000)
plt.show()