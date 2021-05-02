import CIFARloader as data
import matplotlib.pyplot as plt
import numpy as np


def extend_pixels(data):
    final = []
    for image in data:
        final += image
    return final


def calc_errors(image):
    error_image = [image[0]]
    for pixel in range(1,len(image)):
        diff = image[pixel] - image[pixel - 1]
        error_image.append(diff)
    return error_image


data.show_image(calc_errors(data.load_gray_data(10)[3]))
# print(data.load_gray_data(10)[3])
# print(calc_errors(data.load_gray_data(10)[3]))

# print(extend_pixels(calc_errors(data.load_gray_data(1))))
print("mean: " + str(np.mean(calc_errors(data.load_gray_data(10)[3]))))
print("std. dev: " + str(np.std(calc_errors(data.load_gray_data(10)[3]))))
plt.hist(calc_errors(data.load_gray_data(10)[3]))
plt.title('Pixel Intensity Distribution')
plt.xlabel('pixel intensity')
plt.ylabel('number of pixels')
plt.show()
