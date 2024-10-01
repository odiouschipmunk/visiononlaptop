import cv2
import numpy as np

# Define the dimensions of the image
height = 340
width = 640

# Create a white image (all pixel values set to 255)
white_image = np.ones((height, width, 3), dtype=np.uint8) * 255

# Save the image to a file
cv2.imwrite('white_image.png', white_image)

# Display the image
cv2.imshow('White Image', white_image)
cv2.waitKey(0)
cv2.destroyAllWindows()