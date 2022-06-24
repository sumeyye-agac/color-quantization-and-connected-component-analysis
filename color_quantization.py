from PIL import Image
from matplotlib import pyplot as plt
import math
import numpy as np

# Initializations
K = 2 # 2, 4, 8, 16 and 32
manual_selection = False
iteration_number = 10
input_path = '1.jpg'
output_path = '1_K' + str(K) + '.jpg'
img = Image.open(input_path)

img_RGB = img.convert('RGB')
width, height = img_RGB.size

# Distance calculation function
def distance_calculation(color_one, color_two):
    distance = 0
    for i in range(len(color_one)):
        distance = distance + (int(color_one[i]) - int(color_two[i]))**2
    return math.sqrt(distance)

# K-means function
def k_means(pixel_matrix, color_centers):
    cluster_dictionary = {}
    for i in range(iteration_number):
        print(str(i) + ". Iteration.")
        output_array = np.zeros((height * width, 3))

        for j in range(len(color_centers)):
            cluster_dictionary[j] = []
        for k in range(height * width):
            distance = []
            for center in color_centers:
                distance.append(distance_calculation(pixel_matrix[k], center))
            nearest_center = distance.index(min(distance))
            nearest_center_rgb = color_centers[nearest_center]
            output_array[k] = nearest_center_rgb
            cluster_dictionary[nearest_center].append(pixel_matrix[k])

        for element in cluster_dictionary:
            color_centers[element] = np.mean(cluster_dictionary[element], axis=0)

    output_image = output_array.reshape(height, width, 3)
    image = Image.fromarray(output_image.astype('uint8'), 'RGB')
    image.save(output_path)

# Color quantization function
def quantize(img, K):
    image_array = np.array(img)
    pixel_matrix = image_array.reshape((width * height, 3))

    selected_color_centers = [] # x y coordinates
    color_centers = []          # RGB colors

    if manual_selection == True:
        plt.imshow(img)
        selected_color_centers = plt.ginput(K, show_clicks=True)
        plt.close()
    else:
        for i in range(K):
            x_point = np.random.uniform(0, width)
            y_point = np.random.uniform(0, height)
            selected_color_centers.append([x_point, y_point])

    for i in range(K):
        color_centers.append(image_array[int(selected_color_centers[i][1]), int(selected_color_centers[i][0])])

    k_means(pixel_matrix, color_centers)

def main():
    quantize(img_RGB, K)

if __name__ == '__main__':
    main()