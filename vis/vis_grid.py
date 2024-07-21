import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def add_grid_to_image(image_path, M, N, line_width=1, line_color='white'):
    # 定义颜色字典
    color_dict = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
    }

    # 将颜色名称转换为BGR格式
    if line_color in color_dict:
        bgr_color = color_dict[line_color]
    else:
        raise ValueError(f"不支持的颜色: {line_color}")

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请确保路径正确。")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 获取图像尺寸
    height, width, _ = image.shape

    # 创建网格线坐标
    vertical_lines = np.linspace(0, width, N + 1, dtype=int)
    horizontal_lines = np.linspace(0, height, M + 1, dtype=int)

    # 绘制网格线
    for x in vertical_lines:
        image = cv2.line(image, (x, 0), (x, height), color=bgr_color, thickness=line_width)
    for y in horizontal_lines:
        image = cv2.line(image, (0, y), (width, y), color=bgr_color, thickness=line_width)

    # 将图像保存为新文件
    base, ext = os.path.splitext(image_path)
    new_image_path = f"{base}_grid{ext}"
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转换回BGR格式以保存
    cv2.imwrite(new_image_path, image_bgr)

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# 调用函数示例
image_path = '1x2.png'  # 替换为你的图像路径
add_grid_to_image(image_path, M=1, N=2, line_width=12, line_color='white')
