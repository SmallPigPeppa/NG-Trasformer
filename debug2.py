import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
torch.manual_seed(0)
np.random.seed(0)

num_class = 100
temperature = 0.1  # 调整温度系数

# 生成随机值并排序
raw_values = np.random.uniform(size=num_class)
sorted_values = np.sort(raw_values)[::-1]

# 使用温度系数调整能量值
transformed_energy_values = np.exp(sorted_values / temperature)
transformed_energy_values = transformed_energy_values / np.sum(transformed_energy_values)

# 转换为torch tensor
energy_values_tensor = torch.tensor(transformed_energy_values*30, dtype=torch.float32)

# 打印生成的能量值
print(energy_values_tensor)

# 绘制生成的能量值曲线
plt.plot(energy_values_tensor.numpy())
plt.title(f'Sorted Energy Values (Temperature = {temperature})')
plt.xlabel('Index')
plt.ylabel('Energy Value')
plt.show()
