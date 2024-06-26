import torch
import numpy as np
import matplotlib.pyplot as plt
num_class=100
energy_values = torch.tensor(np.random.normal(size=num_class), dtype=torch.float32)
print(energy_values)

# 绘制生成的能量值曲线
plt.plot(energy_values.numpy())
plt.title(f'Sorted Energy Values (random)')
plt.xlabel('Index')
plt.ylabel('Energy Value')
plt.savefig('demo.jpg')
plt.show()

#
# # Step 1: Generate random values between 0 and 1
# random_values = np.random.rand(num_class)
#
# # Step 2: Sort the values in descending order
# sorted_values = np.sort(random_values)[::-1]
#
# # Step 3: Normalize the sorted values to sum to 1
# normalized_values = sorted_values / np.sum(sorted_values)
#
# # Convert to torch tensor
# energy_values = torch.tensor(normalized_values, dtype=torch.float32)
#
# print(energy_values)
#
# plt.plot(energy_values.numpy(), marker='o')
# plt.title('Energy Values Distribution')
# plt.xlabel('Class Index')
# plt.ylabel('Energy Value')
# plt.grid(True)
# plt.show()
