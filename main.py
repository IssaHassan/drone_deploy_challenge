import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import math

PATTERN_PATH = 'photos/pattern.png'
IMAGE_PATH = 'photos/img_6727.jpg'
JPG = '.jpg'


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
		
		#get matched keypoints for pattern image and iphone image 
		self.pattern_kp_matches,self.iphone_kp_matches = self.get_kp_matches()

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
		
		#returns list of keypoints in iphone image and pattern that mathced with each other
		
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
		
	def get_iphone_pt_matches(self):
		#return list of points(tuples of x,y) of matched keypoints
		iphone_matches = []
		
		for kp in self.iphone_kp_matches:
			iphone_matches.append(kp.pt)
		
		return iphone_matches
		
		
		
	def show_kp_matches(self):
		#show plot of keypoints in the iphone image and pattern image that matched with each other
		
		self.show_keypoints(self.pattern_kp_matches,self.pattern_img)
		self.show_keypoints(self.iphone_kp_matches,self.iphone_img)
		

class Location:
	
	def __init__(self, keypoints):
		
		#Accepts keypoints as a list of tuples of size two, which contain (x,y) coordinates 
		#of each keypoint
		self.kp = keypoints
		self.max_kp_pair = self.get_max_kp_distance_pair()
		self.max_kp_distance = self.get_max_kp_distance()
		
	
	def distance_squared(self, points):
		"""
		Accepts two points and returns the squared distance between them,
		there is no reason to compare square roots, because the difference will be the same
		"""	
		p1,p2 = points
		return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)
	
	def get_max_kp_distance_pair(self):
		"""
		Returns the keypount pair(as a tuple) whcih are the furthest apart form each other,
		used to calculate the max distance between pairs.
		"""
		return max(itertools.combinations(self.kp, 2), key=self.distance_squared)
	
	def get_max_kp_distance(self):
		#returns max distance between keypoints
		
		return math.sqrt(self.distance_squared(self.max_kp_pair))
		
	def remove_kp_pair(self, pair):
		p1 = pair[0]
		p2 = pair[1]
		
		for x in self.kp:
			if x[0] == p1[0] or x[0] == p1[1] or x[1] == p1[0] or x[1] == p1[1]:
				if p1 in self.kp:
					self.kp.remove(p1)

			if x[0] == p2[0] or x[0] == p2[1] or x[1] == p2[0] or x[1] == p2[1]:
				if p2 in self.kp:
					self.kp.remove(p2)
		

		
	def get_best_kp_dist_pairs(self):
		"""
		temp = self.kp
		saved = self.kp
		"""
		max_distances = []
		max_kp_values = []

		max_kp_values.append(self.max_kp_pair)
		max_distances.append(self.max_kp_distance)
		self.remove_kp_pair(self.max_kp_pair)
		
		for _ in range(5):
			pt = self.get_max_kp_distance_pair()
			self.max_kp_pair = pt
			max_kp_values.append(pt)
			max_distances.append(self.get_max_kp_distance())
			self.remove_kp_pair(pt)
		
		return max_distances, max_kp_values
		
		
		
def show_all_matches(matches):
	for m in matches:
		m.show_kp_matches()
		
def main():

	m = Match(IMAGE_PATH)
	#m.show_kp_matches()
	l = Location(m.get_iphone_pt_matches())
	print(l.get_best_kp_dist_pairs()[0])
	"""
	l = Location(m.get_iphone_pt_matches())
	print(l.get_max_kp_distance())
	#m.show_kp_matches()
	
	"""
	"""
	l = Location(m.get_iphone_pt_matches())
	print(l.get_max_kp_distance_pair())
	#print(m.get_iphone_pt_matches())
	"""
	
if __name__ == "__main__":
	main()
	
