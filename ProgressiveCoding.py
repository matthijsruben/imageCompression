import CIFARloader as data
import numpy as np
import math


# get the pixel of an image using x and y coordinates that range between 0 and 31
# note that x goes from left to right (0 to 31) and y from top to bottom (0 to 31)
def pix(image, x, y):
    return image[y][x]


# check if the pixel is in the image (are the x and y coordinates coordinates present in the image)
def pixel_in_image(pixel, n):
    return 0 <= pixel[0] < n and 0 <= pixel[1] < n


# calculates the distance (in either x- or y direction) between pixels of two different successive levels
# (of an nxn image). Note that always: parent_level < child_level
def dist_levels(parent_level, n):
    if parent_level % 2 == 0:
        dist = n * 2 ** (-0.5 * parent_level)
    else:
        dist = n * 2 ** (-0.5 * parent_level - 0.5)
    return dist


def find_diamond(pixel, dist):
    p1 = (int(pixel[0] + dist), int(pixel[1]))
    p2 = (int(pixel[0] - dist), int(pixel[1]))
    p3 = (int(pixel[0]), int(pixel[1] + dist))
    p4 = (int(pixel[0]), int(pixel[1] - dist))
    return p1, p2, p3, p4


def find_square(pixel, dist):
    p1 = (int(pixel[0] + dist), int(pixel[1] + dist))
    p2 = (int(pixel[0] + dist), int(pixel[1] - dist))
    p3 = (int(pixel[0] - dist), int(pixel[1] + dist))
    p4 = (int(pixel[0] - dist), int(pixel[1] - dist))
    return p1, p2, p3, p4


# find the pixels of the next level around this given pixel
def calc_pixs(level, pixel, n):
    dist = dist_levels(level, n)
    if level % 2 == 0:
        p1, p2, p3, p4 = find_diamond(pixel, dist)
    else:
        p1, p2, p3, p4 = find_square(pixel, dist)
    return p1, p2, p3, p4


# find all pixels of the next level by recursion
def check_surroundings(found, level, pixel, n):
    found.add(pixel)
    p1, p2, p3, p4 = calc_pixs(level, pixel, n)
    if pixel_in_image(p1, n) and p1 not in found:
        found.add(p1)
        found = found.union(check_surroundings(found, level, p1, n))
    if pixel_in_image(p2, n) and p2 not in found:
        found.add(p2)
        found = found.union(check_surroundings(found, level, p2, n))
    if pixel_in_image(p3, n) and p3 not in found:
        found.add(p3)
        found = found.union(check_surroundings(found, level, p3, n))
    if pixel_in_image(p4, n) and p4 not in found:
        found.add(p4)
        found = found.union(check_surroundings(found, level, p4, n))
    return found


def find_extension(center_pixel, corner_pixel, dist):
    corpx = corner_pixel[0]
    corpy = corner_pixel[1]
    cenpx = center_pixel[0]
    cenpy = center_pixel[1]
    signx = 2*(corpx > cenpx)-1
    signy = 2*(corpy > cenpy)-1
    if corpy == cenpy:
        extension = {(int(corpx + signx * dist), int(corpy - dist)), (int(corpx + signx * 2 * dist), corpy),
                     (int(corpx + signx * dist), int(corpy + dist))}
    elif corpx == cenpx:
        extension = {(int(corpx - dist), int(corpy + signy * dist)), (corpx, int(corpy + signy * 2 * dist)),
                     (int(corpx + dist), int(corpy + signy * dist))}
    else:
        extension = {(int(corpx + signx * dist * 2), corpy), (corpx, int(corpy + signy * dist * 2)),
                     (int(corpx + signx * dist * 2), int(corpy + signy * dist * 2))}
    return extension


# arguments: pixel to be predicted, level in which it is predicted (so parent_level!)
# returns the 4x4 grid coordinates used to predict pixel
def find_4x4(pixel, level, n):
    dist = dist_levels(level, n)
    if level % 2 == 0:
        corners = set(find_diamond(pixel, dist))
    else:
        corners = set(find_square(pixel, dist))
    extensions = set()
    for corner_pixel in corners:
        extensions = extensions.union(find_extension(pixel, corner_pixel, dist))
    return corners.union(extensions)


# creates dictionary with (key, value) = (level, [(x1,y1), (x2,y2), ..., (xn, yn)])
# the value is the set of known pixels for a given level, for an nxn image
def create_level_dict(n):
    level_dict = {}
    for level in range(0, int(2 * math.log(n, 2))):
        level_dict[level+1] = check_surroundings(set(), level, (0, 0), n)
    # last level (11) is basically the complete image
    level_dict[11] = set()
    for x in range(0, n):
        for y in range(0, n):
            level_dict[11].add((x, y))
    return level_dict


# def get_pixel_value(pixel, values, n):
#     value = 128
#     if pixel_in_image(pixel, n):
#         value = values[pixel[1]][pixel[0]]
#     return value


def predict_pixel(pixel, known_values, level, n):
    grid = find_4x4(pixel, level, n)
    grid_values = []
    for grid_pixel in grid:
        if pixel_in_image(grid_pixel, n):
            grid_values.append(known_values[grid_pixel[1]][grid_pixel[0]])
        # grid_values.append(get_pixel_value(grid_pixel, known_values, n))
    return sum(grid_values) / len(grid_values)


def calc_values_of_next_level(known_values, parent_level, n, level_dict):
    new_values = known_values
    pixel_coordinates = level_dict.get(parent_level+1).difference(level_dict.get(parent_level))
    for pixel in pixel_coordinates:
        new_values[pixel[1]][pixel[0]] = predict_pixel(pixel, known_values, parent_level, n)
    return new_values


# predict all pixels of an image based on a 2D array of known pixels (values)
def get_all_values(start_level, values, n, level_dict):
    for level in range(start_level, int(2 * math.log(n, 2)) + 1):
        values = calc_values_of_next_level(values, level, n, level_dict)
    return values


def find_starting_values(image, start_level, n, level_dict):
    values = np.zeros((n, n))
    known_pixels = level_dict.get(start_level)
    for p in known_pixels:
        values[p[1]][p[0]] = image[p[1]][p[0]]
    return values


def predict_image_at_starting_level(image, start_level, n):
    level_dict = create_level_dict(n)
    known_values = find_starting_values(image, start_level, n, level_dict)
    predicted_image = get_all_values(start_level, known_values, n, level_dict)
    return predicted_image


import matplotlib.pyplot as plt

imageArray = data.load_gray_data(1000)

all_maxes = []
for img in imageArray:
    image = np.array(img).reshape(32, 32)
    maxes = []
    for level in range(1, 12):
        pred_image = predict_image_at_starting_level(image, level, 32)
        # print("max of level " + str(level) + str(max(np.array(pred_image - image).flatten()))) # data.show_image(pred_image - image)
        # print("LEVEL " + str(level))
        maxes.append(max(np.array(pred_image - image).flatten()))
        # print(max(np.array(pred_image - image).flatten()))
        # print(np.array(pred_image - image).flatten())
    # data.show_image(image)
    # print(maxes)
    all_maxes.append(maxes)
a1 = np.array([[1,2,3],
               [10, 20, 30],
               [12, 13, 14]])
a2 = np.array([[10, 20, 30],
               [1,2,3],
               [12, 13, 14]])
# print(max(map(max, a1)))
# print(a1.flatten())

ar1 = [10, 8, 4]
ar2 = [12, 9, 5]
ar3 = [11, 10, 6]
ars = [ar1, ar2, ar3]

print([np.mean(col) for col in zip(*all_maxes)])


# plt.hist(np.array((pred_image - image)).flatten())
# plt.title('Pixel Intensity Distribution')
# plt.xlabel('pixel intensity')
# plt.ylabel('number of pixels')
# plt.show()
