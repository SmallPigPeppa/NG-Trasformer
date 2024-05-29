import matplotlib.pyplot as plt

# Data
ratio = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
acc = [85.10, 85.09, 85.11, 85.09, 85.04, 85.05, 84.92, 84.72, 83.93, 78.27]

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plotting with ax.plot
ax.plot(ratio, acc, marker='o', linestyle='-')

# Setting labels and title with larger font sizes
ax.set_xlabel('Ratio', fontsize=20)
ax.set_ylabel('Accuracy (%)', fontsize=20)
# ax.set_title('Accuracy vs. Ratio', fontsize=20)

# Inverting x-axis
ax.invert_xaxis()

# Setting ticks with larger font sizes
ax.tick_params(axis='both', which='major', labelsize=16)

# Adding grid
ax.grid(True)

plt.tight_layout()
plt.savefig('demo.jpg')
# Show plot
plt.show()
