import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
torch.manual_seed(0)
np.random.seed(0)

num_class = 100
temperature = 0.3  # 调整温度系数

# 生成服从正态分布的随机值
energy_values = np.random.normal(loc=0, scale=1, size=num_class)

# 使用温度系数调整能量值
transformed_energy_values = np.exp(energy_values / temperature)
transformed_energy_values = transformed_energy_values / np.sum(transformed_energy_values)

# 转换为torch tensor
energy_values_tensor = torch.tensor(transformed_energy_values, dtype=torch.float32)

# 排序
sorted_energy_values = torch.sort(energy_values_tensor, descending=True).values

# 打印生成的能量值
print(sorted_energy_values)

# 绘制生成的能量值曲线
plt.plot(sorted_energy_values.numpy())
plt.title(f'Sorted Energy Values (Temperature = {temperature})')
plt.xlabel('Index')
plt.ylabel('Energy Value')
plt.show()
