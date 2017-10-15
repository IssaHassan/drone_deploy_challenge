import cv2
import numpy as np
from matplotlib import pyplot as plt

PATTERN_PATH = 'photos/pattern.png'
IMAGE_PATH = 'photos/img_67'
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
		
		self.kp = keypoints
		
	
	def distance_squared(self, points):
		"""
		Accepts two points and returns the squared distance between them,
		there is no reason to compare square roots, because the difference will be the same
		"""	
		p1,p2 = points
		return math.sqrt((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)
	
	def greatest_kp_distance(self):
		
		
		

def show_all_matches(matches):
	for m in matches:
		m.show_kp_matches()
		
def main():
	
	matches = []
	
	for x in range(7):
		f = IMAGE_PATH+str(19+x)+JPG
		matches.append(Match(f))
	
	show_all_matches(matches)
	
	"""
	m = Match(IMAGE_PATH)

	print(m.get_iphone_pt_matches())
	"""
	
if __name__ == "__main__":
	main()
	
