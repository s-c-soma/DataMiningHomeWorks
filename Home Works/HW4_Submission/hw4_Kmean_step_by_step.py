#!/usr/bin/env python

import math

# list of 1st 20 primes
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

class Point:
	def __init__(self, x=0, y=0):
        	self.x = x
        	self.y = y
	def assign(self, _x, _y):
		self.x = _x
		self.y = _y
	def show(self):
		print("Point: x = %s, y = %s\n" % (self.x, self.y))

def euclidean_dist(a, b):
	return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def generate_points():
	mPoints = []
	for i in range(0, len(primes), 2):
		mPoints.append(Point(primes[i], primes[i + 1]))
	#for point in mPoints:
	#	point.show()
	#print len(mPoints)
	return mPoints

# returns clusters: list of cluster
# each cluster: list of points
def find_clusters(points, centroids):
	clusters = []
	k = len(centroids)
	for i in range(k):
		clusters.append([])

	for point in points:
		min_dist = 1000000000.0	#INF
		idx = -1
		for j in range(k):
			now_dist = euclidean_dist(point, centroids[j])
			if (now_dist < min_dist):
				min_dist = now_dist
				idx = j
		clusters[idx].append(point)

	return clusters
		
def compute_centroids(clusters):
	centroids = []
	for cluster in clusters:
		sum_x = 0
		sum_y = 0
		for pt in cluster:
			sum_x += pt.x
			sum_y += pt.y
		centroids.append(Point(sum_x / len(cluster), sum_y / len(cluster)))
	return centroids

def find_point(p, clusters):
	v = len(clusters)
	for i in range(v):
		for pt in clusters[i]:
			if pt.x == p.x and pt.y == p.y:
				return i
	return -1 

def are_same_clusters(A, B):
	if len(A) != len(B):
		return False
	
	A.sort(key = len)
	B.sort(key = len)

	for cl in A:
		found = -1
		for pt in cl:
			idx = find_point(pt, B)
			if found == -1:
				found = idx
			elif found != idx:
				return False
	return True

def k_means_clustering(points, centroids):
	prev_clusters = []
	itr = 0
	while True:
		itr += 1
		print("***Iteration %s ***" % itr)

		new_clusters = find_clusters(points, centroids)
		v = len(new_clusters)
		for i in range(v):
			print("cluster %s", i)
			for pt in new_clusters[i]:
				pt.show()

		new_centroids = compute_centroids(new_clusters)
		print("Centroids:")
		for pt in new_centroids:
			pt.show()

		if are_same_clusters(prev_clusters, new_clusters) == True:
			break;
		points = []
		for cl in new_clusters:
			for pt in cl:
				points.append(pt)
		centroids = new_centroids
		prev_clusters = new_clusters

	return prev_clusters

def inter_cluster_distance(A, B):
	sum = 0.0
	for pa in A:
		for pb in B:
			sum += euclidean_dist(pa, pb)
	return sum / (len(A) * len(B))

def intra_cluster_distance(A):
	sum = 0.0
	n = len(A)
	for i in range(n):
		for j in range(n):
			if i != j:
				sum += euclidean_dist(A[i], A[j])

	return sum / (n * (n - 1))

def main():
	mPoints = generate_points()

	centroids = [Point(2, 3), Point(5, 7)]

	clusters = k_means_clustering(mPoints, centroids)

	print("Inter-cluster distance: %s" %inter_cluster_distance(clusters[0], clusters[1]))

	n = len(clusters)
	for i in range(n):
		print("Intra-cluster distance for cluster %s: %s" % (i, intra_cluster_distance(clusters[i])))
	

main()
