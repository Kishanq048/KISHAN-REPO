import numpy as np
import cv2 as cv

def read_image_from_text_file(file_path):
    with open(file_path, 'r') as file:
        
        lines = file.readlines()

    data = []
    for line in lines:
        row = list(map(int, line.split()))
        data.append(row)

    image_array = np.array(data, dtype=np.int32)  

    image_array[image_array < 0] = 0
    image_array[image_array > 255] = 255

    image_array = image_array.astype(np.uint8)

    return image_array

file_path = 'output_image2.txt' 
image_array = read_image_from_text_file(file_path)

cv.imwrite('output_image2.jpg', image_array)
cv.imshow('Image', image_array)
cv.waitKey(0)
cv.destroyAllWindows()
