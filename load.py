import cv2 as cv
import argparse
import sys

# Set up argument parser
parser = argparse.ArgumentParser(description='Process an image and save its grayscale values to a text file.')
parser.add_argument('image_path', type=str, help='Path to the image file')
args = parser.parse_args()

# Read the image from the provided path
file = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)

if file is None:
    print(f"Error: Unable to open image file {args.image_path}")
    sys.exit(1)

height = file.shape[0]
width = file.shape[1]

with open('text1.txt', 'w') as f:

    f.write(f"{height} {width}\n")
    for i in range(height):

        for j in range(width):

            f.write(f"{file[i, j]} ")

        f.write("\n")

cv.imshow('Image', file)
print("Grayscale values have been written to text1.txt.")
