import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'birds1.jpg'
image_name = 'birds1.jpg'
if image_name[-4:] == '.png':
    img = cv2.imread(image_path)
    height, width,  channels = img.shape
else:
    img = cv2.imread(image_path, 0)
    height, width = img.shape

foreground_pixel = 0
componentQueue = [] # x,y coordinates of unlabelled foreground pixels


# Apply different thresholds, morphological operations
#  and some other thresholding techniques for preprocessing
#  based on the image name information.
def preprocessingOfImage(img, image_name):

    if image_name == 'birds1.jpg' or image_name == 'birds2.jpg':
        ret, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        binary_image_erosion = cv2.erode(binary_image, kernel, iterations=1)
        preprocessed_image = cv2.dilate(binary_image_erosion, kernel, iterations=1)

    if image_name == 'birds3.jpg':
        ret, binary_image = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        binary_image_erosion = cv2.erode(binary_image, kernel, iterations=1)

        kernel = np.ones((4, 4), np.uint8)
        preprocessed_image = cv2.dilate(binary_image_erosion, kernel, iterations=1)

    if image_name == 'dice5.png':
        hsv_low = np.array([0, 0, 0])
        hsv_high = np.array([10, 10, 10])
        # hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img, hsv_low, hsv_high)

        kernel = np.ones((3, 3), np.uint8)
        image_dilatation = cv2.dilate(mask, kernel, iterations=1)
        image_erosion = cv2.erode(image_dilatation, kernel, iterations=1)

        kernel = np.ones((2, 2), np.uint8)
        image_erosion2 = cv2.erode(image_erosion, kernel, iterations=1)

        kernel = np.ones((6, 6), np.uint8)
        image_dilatation2 = cv2.dilate(image_erosion2, kernel, iterations=2)

        ret, preprocessed_image = cv2.threshold(image_dilatation2, 127, 255, cv2.THRESH_BINARY_INV)

    if image_name == 'dice6.png':
        hsv_low = np.array([0, 0, 0])
        hsv_high = np.array([10, 10, 6])
        # hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img, hsv_low, hsv_high)

        kernel = np.ones((4, 4), np.uint8)
        image_dilatation = cv2.dilate(mask, kernel, iterations=1)

        kernel = np.ones((3, 3), np.uint8)
        image_erosion = cv2.erode(image_dilatation, kernel, iterations=1)

        kernel = np.ones((2, 2), np.uint8)
        image_erosion2 = cv2.erode(image_erosion, kernel, iterations=1)

        kernel = np.ones((6, 6), np.uint8)
        image_dilatation2 = cv2.dilate(image_erosion2, kernel, iterations=1)

        ret, preprocessed_image = cv2.threshold(image_dilatation2, 127, 255, cv2.THRESH_BINARY_INV)

    if image_name == 'dice6_2.png':
        hsv_low = np.array([0, 0, 0])
        hsv_high = np.array([10, 10, 0])
        # hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img, hsv_low, hsv_high)

        kernel = np.ones((4, 4), np.uint8)
        image_dilatation = cv2.dilate(mask, kernel, iterations=1)

        kernel = np.ones((3, 3), np.uint8)
        image_erosion = cv2.erode(image_dilatation, kernel, iterations=1)

        kernel = np.ones((2, 2), np.uint8)
        image_erosion2 = cv2.erode(image_erosion, kernel, iterations=1)

        kernel = np.ones((8, 8), np.uint8)
        image_dilatation2 = cv2.dilate(image_erosion2, kernel, iterations=1)

        kernel = np.ones((2, 2), np.uint8)
        image_dilatation3 = cv2.dilate(image_dilatation2, kernel, iterations=1)

        ret, preprocessed_image = cv2.threshold(image_dilatation3, 127, 255, cv2.THRESH_BINARY_INV)

    plt.imshow(preprocessed_image, cmap='gray')
    plt.show()

    return preprocessed_image

# Connected neighbours searching and labelling function
def connectedNeighboursSearch(new_image, componentCounter):
    connected_pixels = [[-1,-1], [-1,0], [-1,1],
                        [0, -1], [0, 0], [0, 1],
                        [1, -1], [1, 0], [1, 1]]
    element_of_queue = componentQueue.pop(0)
    for neighbour in connected_pixels:
        current_y = element_of_queue[0] + neighbour[0]
        current_x = element_of_queue[1] + neighbour[1]
        if new_image[current_y][current_x] == foreground_pixel:
            if 0 <= current_y < height and 0 <= current_x < width:
                new_image[current_y][current_x] = componentCounter # labelling
                componentQueue.append([current_y, current_x])
                connectedNeighboursSearch(new_image, componentCounter)

# Connected component counting function
def countConnectedComponents(img, image_name):

    new_image = preprocessingOfImage(img, image_name)

    row_number = 0
    componentCounter = 0

    while row_number < height:
        column_number = 0
        while column_number < width:
            if new_image[row_number][column_number] == foreground_pixel: # a pixel of new component is detected
                componentCounter += 1
                new_image[row_number][column_number] = componentCounter  # labelling
                componentQueue.append([row_number,column_number])
                connectedNeighboursSearch(new_image, componentCounter)
            column_number += 1
        row_number += 1
    print("Number of components in the image is ", componentCounter)

def main():
    countConnectedComponents(img, image_name)

if __name__ == '__main__':
    main()
