import matplotlib.pyplot as plt
import numpy as np

# Data
categories = [
    "Repository Link",
    "Model Code",
    "Baseline Code",
    "Hyperparameter-tuning Code",
    "Preprocessing Code",
    "Supplementary Info."
]
ticks = ["✓", "X"]

# Values for each segment (✓ and X)
data_check = [41, 35, 9, 5, 25, 36]
data_x = [15, 1, 27, 31, 11, 20]

# Position of bars
y_pos = np.arange(len(categories))

# Create horizontal stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Bars
ax.barh(y_pos, data_check, color="lightgray", label="✓")
ax.barh(y_pos, data_x, left=data_check, color="darkgray", label="X")

# Add labels
ax.set_yticks(y_pos)
ax.set_yticklabels(categories)
ax.set_xlabel("Counts")
ax.set_title("Availability of Components")
ax.legend(loc="upper right")

# Add text annotations for each bar
for i in range(len(categories)):
    ax.text(data_check[i] / 2, i, str(data_check[i]), va="center", ha="center", color="black")
    ax.text(data_check[i] + data_x[i] / 2, i, str(data_x[i]), va="center", ha="center", color="white")

# Adjust layout
plt.tight_layout()
plt.show()