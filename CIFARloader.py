import pickle
import matplotlib.pyplot as plt
import numpy as np
import time


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# replaces each image-array [r1, r2, ..., rn, g1, g2, ..., gn, b1, b2, ..., gn]
# by                        [(r1, r2, ..., rn), (g1, g2, ..., gn), (b1, b2, ..., bn)]
def split(data, n):
    n1 = int(len(data[0]) / n)
    n2 = int(2 * n1)
    return list(map(lambda x: list(zip(*[x[:n1], x[n1:n2], x[n2:]])), data))


# 3 options for rgb-grayscale conversion:
# 1: Y = sum(r,g,b) / 3
# 2: Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
# 3: Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
def image_to_grayscale(image):
    return list(map(lambda col: int(0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2]), image))


def list_to_grayscale(data):
    return list(map(lambda image: image_to_grayscale(image), data))


def show_image(gray_image):
    plt.imshow(np.array(gray_image).reshape(32, 32), cmap='gray')
    plt.show()


# n is the first n images you want to load. For CIFAR-100, max(n)=50000
def load_gray_data(n):
    dict = unpickle('../cifar-100-python/train')
    data = dict.get(b'data')[:n]
    return list_to_grayscale(split(data, 3))



# start = time.time()
# print(load_gray_data())
# end = time.time()
# print("Elapsed Time: " + str(end - start))  # the process takes roughly 7 min.
