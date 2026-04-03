import matplotlib.pyplot as plt
import re
import sys

steps = []
text_loss = []
mel_loss = []
mel_top1 = []

with open(sys.argv[1],"r") as f:
    for line in f:
        segs = line.strip().split()
        if len(segs)<1 or segs[0] != "[Val]":
            continue            
        steps.append(float(segs[2].split("=")[1]))
        text_loss.append(float(segs[3].split("=")[1]))
        mel_loss.append(float(segs[4].split("=")[1]))
        mel_top1.append(float(segs[5].split("=")[1]))
        print(steps[-1],text_loss[-1],mel_loss[-1],mel_top1[-1])

# The log data you provided
#[Train] epoch=1 step=1 text_loss=2.5507 mel_loss=5.9433 mel_top1=0.0543 lr=1.00e-08

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Loss Curves
ax1.plot(steps, text_loss, label='Text Loss', color='royalblue', marker='o')
ax1.plot(steps, mel_loss, label='Mel Loss', color='darkorange', marker='s')
ax1.set_ylabel('Loss Value')
ax1.set_title('Valid Loss Curves')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot Accuracy (Top1)
ax2.plot(steps, mel_top1, label='Mel Top1 Acc', color='forestgreen', marker='^')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Accuracy')
ax2.set_title('Mel Prediction Accuracy (Top1)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
