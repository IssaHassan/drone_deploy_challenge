import cv2
import numpy as np
from matplotlib import pyplot as plt

PATTERN_PATH = 'photos/pattern.png'
IMAGE_PATH = 'photos/img_6727.jpg'


class Match:
	
	def __init__(self, filename):
		
		self.pattern_img = cv2.imread(PATTERN_PATH,0)
		self.iphone_img = cv2.imread(filename)
		
		#initialize ORB (Oriented FAST and Rotated BRIEF) feature and descriptor detector
		self.orb = cv2.ORB_create()
		
		# find keypoints and descriptors for iphone image and pattern image.
		self.pattern_kp, self.pattern_desc = self.orb.detectAndCompute(self.pattern_img,None)
		self.iphone_kp, self.iphone_desc = self.orb.detectAndCompute(self.iphone_img,None)
		
		# initialize brute force matcher
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		self.matches = self.get_matches()
		

	def get_keypoints(self,image):
		
		#returns list of keypoints form the image.
		return self.orb.detect(image,None)

	def show_keypoints(self,kp,image):
		
		#draw markers on image to show keypoints

		image_kp = cv2.drawKeypoints(image,kp,255, 0)
		plt.imshow(image_kp)
		plt.show()
	
	def get_matches(self):
		
		#match image and pattern descriptors
		temp = self.bf.match(self.pattern_desc,self.iphone_desc)
		
		#return sorted list of DMatch objects, matches with least distance are first in the list
		return sorted(temp,key = lambda d:d.distance)
		
	def get_kp_matches(self):
		
		#returns list of keypoint positions in iphone image and pattern that mathced with each other
		
		pattern_kp_matches = []
		iphone_kp_matches = []
		
		for m in self.matches:
			
			#get index in total keypoints for each matched index 
			#queryIdx refers to the first descriptor passed to bf.match (pattern), 
			#trainIdx refers to the second
			
			pattern_idx = m.queryIdx
			iphone_idx = m.trainIdx
		
			#append each keypoint that matched to the list of pattern matches and iphone matches
			pattern_kp_matches.append(self.pattern_kp[pattern_idx])
			iphone_kp_matches.append(self.iphone_kp[iphone_idx])
		
		return pattern_kp_matches, iphone_kp_matches
		
		
	def show_kp_matches(self):
		#show plot of keypoints in the iphone image and pattern image that matched with each other
		
		pattern_kp_matches = self.get_kp_matches()[0]
		iphone_kp_matches = self.get_kp_matches()[1]
		
		self.show_keypoints(pattern_kp_matches,self.pattern_img)
		self.show_keypoints(iphone_kp_matches,self.iphone_img)
		
		

def main():
	
	m = Match(IMAGE_PATH)

	m.show_kp_matches()
	
	
if __name__ == "__main__":
	main()
	
