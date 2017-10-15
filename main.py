import cv2
import numpy as np
from matplotlib import pyplot as plt

PATTERN_PATH = 'photos/pattern.png'
IMAGE_PATH = 'photos/img_6727.jpg'

#read in pattern and iphone image in grayscale
pattern = cv2.imread(PATTERN_PATH,0)
img = cv2.imread(IMAGE_PATH,0)

class Match:
	
	def __init__(self, filename):
		
		self.iphone_img = cv2.imread(filename)
		self.orb = cv2.ORB_create()

	def get_keypoints(self,image):
		
		#returns list of keypoints form the image.
		return self.orb.detect(image,None)

	def print_keypoints(self,kp,image):
		
		"""
			draw markers on iphone image to show keypoints
		"""
		image_kp = cv2.drawKeypoints(image,kp,255, 0)
		plt.imshow(image_kp)
		plt.show()

def main():
	
	m = Match(IMAGE_PATH)
	m.print_keypoints(m.get_keypoints(pattern),pattern)
	m.print_keypoints(m.get_keypoints(m.iphone_img),m.iphone_img)
	
	
if __name__ == "__main__":
	main()
	
