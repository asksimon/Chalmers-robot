import numpy as np
from cv2 import cv2 as cv

img = cv.imread('Images/Victor.png')

# Convert to gray scale
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Invert the image
inv_gray_image = 255 - gray_image

# Blur the Image
Blurred_image = cv.GaussianBlur(inv_gray_image,(21,21),0)

# invert the blurred image
inv_blur = 255 - Blurred_image

sketch = cv.divide(gray_image, inv_blur, scale=256)

# Edge Cascade
canny = cv.Canny(img, )
imagem = cv.bitwise_not(canny)
cv.imshow('canny edges', imagem)



# Show the images
cv.imshow('Image', img)
#cv.imshow('Gray image', gray_image)
cv.imshow('Sketch', sketch)

cv.waitKey(0)
cv.destroyAllWindows()