import numpy as np
from PIL import Image


def get_a_sample(x, y):
    for i in range(12):  # Changed from 11 to 12 for 12 characters
        for j in range(60):
            # m = np.random.randint(1, 13)
            # n = np.random.randint(1, 601)
            address = f'train/{i+1}/{j+1}.bmp'
            image = Image.open(address)
            # 将图片调整为28*28大小
            image = image.resize((28, 28))
            # 将图片转换为灰度图像
            image = image.convert('L')
            # 将图片数据转换为NumPy数组
            image_array = np.array(image)
            # 将图片数组展平为1*784的向量
            vector_1x784 = image_array.flatten().reshape(1, 784)/255
            x.append(vector_1x784)
            vector = np.zeros((1, 12))
            vector[0][i] = 1
            y.append(vector)


