import numpy as np
import cv2
from matplotlib import pyplot as plt

PATTERN_PATH = 'photos/pattern.png'
IMAGE_PATH = 'photos/img_6727.jpg'
OBJ_HEIGHT = 330


def get_pattern_location():
	"""
		finds coordinates in the image of the top left corner and bottom right corner of the
		rectangle that contains the pattern
		returns those two coordinates
	"""
	pattern = cv2.imread(PATTERN_PATH,cv2.IMREAD_GRAYSCALE)
	img = cv2.imread(IMAGE_PATH,cv2.IMREAD_GRAYSCALE)
	
	p_width, p_height = pattern.shape[::-1]
	
	res = cv2.matchTemplate(img,pattern,cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	
	top_left = max_loc
	bottom_right = (top_left[0] + p_width, top_left[1] + p_height)
	
	return [top_left,bottom_right]

def get_distance(xy):
	"""
		returns the average distance between two points, xy passed in as a list of size 2
		
	"""
	point_a = xy[0]
	point_b = xy[1]
	
	distances =  abs(point_a[0] - point_b[0]), abs(point_a[1] - point_b[1])
	return np.mean(distances)	
	
def main():
	print(get_distance(get_pattern_location()))

if __name__== '__main__':
	main()