import cv2
import numpy as np
from matplotlib import pyplot as plt

PATTERN_PATH = 'photos/pattern.png'
IMAGE_PATH = 'photos/img_6727.jpg'

#read in pattern and iphone image in grayscale
pattern = cv2.imread(PATTERN_PATH,0)
img = cv2.imread(IMAGE_PATH,0)

def get_keypoints(image):
	
	orb = cv2.ORB_create()
	#returns list of keypoints form the image.
	return orb.detect(image,None)

def print_keypoints(kp,image):
	
	"""
		draw markers on iphone image to show keypoints
	"""
	image_kp = cv2.drawKeypoints(image,kp,255, 0)
	plt.imshow(image_kp)
	plt.show()

def main():
	print_keypoints(get_keypoints(pattern),pattern)
	print_keypoints(get_keypoints(img),img)
	

if __name__ == "__main__":
	main()
	
