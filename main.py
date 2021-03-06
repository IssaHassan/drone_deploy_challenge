import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import math

PATTERN_PATH = 'photos/pattern.png'
IMAGE_PATH = 'photos/img_6727.jpg'
JPG = '.jpg'

NUM_MAX_DISTANCES = 8

IMAGE_PATTERN_SIZE = 330 #PIXELS
REAL_PATTERN_SIZE = 88 #MM
IPHONE_FOCAL_LENGTH = 35 #MM
SENSOR_HEIGHT = 3.67 #MM

IPHONE_PHOTO_WIDTH = 2448 #PIXELS
IPHONE_PHOTO_HEIGHT = 3264 #PIXELS
IPHONE_CENTER_X = IPHONE_PHOTO_WIDTH/2 #PIXELS
IPHONE_CENTER_Y = IPHONE_PHOTO_HEIGHT/2 #PIXELS
#The width of a pixel of the cameras sensor
PIXEL_WIDTH_SENSOR = 0.0015 #MM 
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
		
		self.h_matrix,self.mask = self.get_homography()

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
		
	def get_homography(self):
		#return homography matrix 
		src_pts = np.float32([ self.pattern_kp[m.queryIdx].pt for m in self.matches ]).reshape(-1,1,2)
		dst_pts = np.float32([ self.iphone_kp[m.trainIdx].pt for m in self.matches ]).reshape(-1,1,2)
		return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	
	
	def intinsic_matrix(self):
		###Note the basis for this function has been taken from https://gist.github.com/sehnkim/1033637
		cx = IPHONE_PHOTO_WIDTH/2
		cy = IPHONE_PHOTO_HEIGHT/2

		#focal length divided by the size of a pixel on the camera sensor will give us our focal length/width in pixels 
		fx = fy = IPHONE_FOCAL_LENGTH/PIXEL_WIDTH_SENSOR
		
		
		m = np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])
		return m
		
	def get_rotation(self):
		return cv2.decomposeHomographyMat(self.h_matrix,self.intinsic_matrix())[1][0]

	def get_euler(self):
		return cv2.RQDecomp3x3(self.get_rotation())[0]
	
class Location:
	
	def __init__(self, keypoints):
		"""
		Accepts keypoints as a list of tuples of size two, which contain (x,y) coordinates 
		of each keypoint
		
		"""
		
		self.kp = keypoints
		self.max_kp_pair = self.get_max_pair()
		self.max_kp_distance = self.get_max_dist()
		
		#final distance to calculate the size of the pattern in the iphone image and the assoicated pair of 
		#points 
		self.dist, self.points = self.get_best_dist_pair() 
		
		self.pattern_size = self.get_pattern_size()

	
	def distance_squared(self, points):
		"""
		Accepts two points and returns the squared distance between them
		
		"""	
		p1,p2 = points
		return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)
	
	def get_max_pair(self):
		"""
		Returns the keypoint pair, in self.kp, with the largest distance between them 

		"""
		return max(itertools.combinations(self.kp, 2), key=self.distance_squared)
	
	def get_max_dist(self):
		"""
		Returns the distance between the two keypoints in self.max_kp_pair 
		"""
		
		return math.sqrt(self.distance_squared(self.max_kp_pair))
		
	def remove_kp_pair(self, pair):
		"""
		Removes a pair of points from self.kp (the set of keypoints) 
		"""
	
	
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
		Returns two lists 
		A list of the 8 largest distances between any two keypoint pairs and 
		A list of tuples representing the 16 associated points (list is size 8) 
		
		After each keypoint is used in a pair
		it is removed to avoid reusing a incorrect keypoint value 
		"""
		
		max_distances = []
		max_kp_values = []

		#initialize the lists with the current values before removing the keypoints from the list and
		#reinitializing the values 
		max_kp_values.append(self.max_kp_pair)
		max_distances.append(self.max_kp_distance)
		self.remove_kp_pair(self.max_kp_pair)
		
		for _ in range(NUM_MAX_DISTANCES):
			
			#get the next pair of points with the maximum distance between them 
			pt = self.get_max_pair()
			self.max_kp_pair = pt
			max_kp_values.append(pt)
			
			#get the distance between the new maximum distance pair 
			max_distances.append(self.get_max_dist())
			self.remove_kp_pair(pt)
		
		return max_distances, max_kp_values
		
	
	def get_best_dist_pair(self):
		"""
		Returns the largest correct distance and the associated pair of points
		"""
		
		dists, vals = self.get_best_kp_dist_pairs()
		dists = dists[::-1]
		vals = vals[::-1]
		
		prev_d, prev_v = dists[0],vals[0]
		
		for d,v in zip(dists,vals):
			if d > 1.41*prev_d:
				return prev_d,prev_v
			prev_d,prev_v = d,v 
		
		return prev_d,prev_v
		
	def get_pattern_size(self):
		"""
		returns the size of the pattern in pixels
		"""

		#self.points gives the two points that give the max distance between any two
		#correct keypoint values 
		p1,p2 = self.points
		
		#find if the difference in x or the difference in y is greater and that will be the size in pixels
		#of the pattern inside the iphone image 
		x = abs(p1[0]-p2[0])
		y = abs(p1[1]-p2[1])
		return max(x,y)		
	
	def get_z(self):
		"""
		returns distance between the camera and the pattern in Z direction in mm
		"""

		return (IPHONE_FOCAL_LENGTH*REAL_PATTERN_SIZE*IMAGE_PATTERN_SIZE)/(self.pattern_size*SENSOR_HEIGHT)
		
	def get_midpoint_x(self):
		"""
		Returns midpoint in x direction between the two main keypoints stored in self.points 
		"""
		x1,_ = self.points[0]
		x2,_ = self.points[1]
		
		
		return (x1+x2)/2
	
	def get_midpoint_y(self):
		
		_,y1 = self.points[0]
		_,y2 = self.points[1]
		return (y1+y2)/2
		
	def get_scale(self):
		"""
		returns real life size of each pixel, or scale of the image in mm/px 
		we have size of the object in real life(mm) and size of the object in the image(px)
		"""

		return REAL_PATTERN_SIZE/self.pattern_size

	
	def get_x(self):
		"""
		returns distance between the camera and the pattern in X direction in mm
		If the camera is to the right of the pattern, it will return a positive value
		If it is to the left a negative value will be returned.
		"""
		
		pattern_dist = IPHONE_CENTER_X - self.get_midpoint_x()
		return pattern_dist*self.get_scale()
		
	
	def get_y(self):
		
		pattern_dist = IPHONE_CENTER_Y - self.get_midpoint_y()
		return  pattern_dist*self.get_scale()
		
		
		
		
		
def show_all_matches(matches):
	for m in matches:
		m.show_kp_matches()
		
def main():

	m = Match(IMAGE_PATH)
	#print(m.get_euler())
	
	l = Location(m.get_iphone_pt_matches())
	
	print('The distance of the image in the z,x, and y directions are: \n')
	print('{0:.1f} mm'.format(l.get_z()))
	print('{0:.1f} mm'.format(l.get_x()))
	print('{0:.1f} mm\n'.format(l.get_y()))
	
	print('The values of the euler angles, yaw, pitch and roll are: \n')
	
	print('{0:.1f} deg'.format(m.get_euler()[0]))
	print('{0:.1f} deg'.format(m.get_euler()[1]))
	print('{0:.1f} deg'.format(m.get_euler()[2]))
	
	
	
	
if __name__ == "__main__":
	main()
	
