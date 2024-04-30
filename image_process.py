import numpy as np
from PIL import Image
import time
from matplotlib import pyplot as plt


def get_a_sample(x, y):
    for i in range(12):  # Changed from 11 to 12 for 12 characters
        for j in range(60):
            # m = np.random.randint(1, 13)
            # n = np.random.randint(1, 601)
            address = f"train/{i+1}/{j+1}.bmp"
            image = Image.open(address)
            # 将图片调整为28*28大小
            image = image.resize((28, 28))
            # 将图片转换为灰度图像
            image = image.convert("L")
            # 将图片数据转换为NumPy数组
            image_array = np.array(image)
            image_array = np.ones([28, 28], dtype=int) * 256 - image_array
            # print(image_array)
            # plt.imshow(image_array, cmap="gray")
            # plt.colorbar()  # 显示颜色条
            # plt.show()
            # time.sleep(10)
            # 将图片数组展平为1*784的向量
            vector_1x784 = image_array.flatten().reshape(1, 28 * 28) / 255
            x.append(vector_1x784)
            vector = np.zeros((1, 12))
            vector[0][i] = 1
            y.append(vector)


def shuffle(x, y):

    state = np.random.get_state()
    np.random.shuffle(x)
    # result:[6 4 5 3 7 2 0 1 8 9]
    np.random.set_state(state)
    np.random.shuffle(y)
