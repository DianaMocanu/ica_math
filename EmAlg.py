from copy import deepcopy
import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
from numpy.linalg import linalg

class Cluster():

    def initializeValues(self, clusters_number, mtx_cov, data_dim):
        self.centroids = np.array([(2*random.random()-1)*15,random.random()*(-17)])
        self.covariance_matrices = np.diag([1 for _ in range(mtx_cov)])
        self.cluster_probability = 1/clusters_number

    def zeroValues(self, m, num_clusters):
        self.centroids = [[0] * m]
        self.covariance_matrices = np.diag([1 for _ in range(m)])
        self.cluster_probability = [0]

class EmAlg():
    def __init__(self, dim_data):
        self.clusters = []
        self.dim_data = dim_data

    def read_data(self, file):
        f = open(file, 'r')
        separator = ' '
        lines = f.readlines()
        points = []
        for line in lines:
            line = line.strip(' \n')
            points.append(np.array([float(x) for x in line.split(separator)]))

        rows = len(points)
        cols = len(points[0])

        return np.array(points), rows, cols


    def plot_data(self, points, cent_clusters):
        # Plot the points along with the clusters
        colors = iter(['red', 'yellow', 'pink', 'green', 'purple', 'black'])
        plt.scatter(points[:,0], points[:,1], s=2.5)
        if(cent_clusters):
            for c in cent_clusters:
                plt.scatter(c.centroids[0], c.centroids[1], color=next(colors), s=200)
        plt.show()

    def initialize_clusters(self, points, num_clusters, m):
        #create n random positioned clusters
        for i in range(num_clusters):
            cluster = Cluster()
            cluster.initializeValues(num_clusters, m,self.dim_data)
            self.clusters.append(cluster)
        self.plot_data(points, self.clusters)

    def init_gamma(self, n, k):
        row = [-1] * n
        w = [np.array(row)] * k
        return w

    def em(self, iterations, points, n, m, num_clusters, isImproved):
        # the em algorithm with a little adjustment based on if we are running it for the first time and we need to initialize the clusters or not
        if not isImproved:
            self.initialize_clusters(points, num_clusters, m)
        for i in range(iterations +1):
            gamma = self.compute_gamma(points, n, num_clusters)
            new_clusters = self.compute_clusters(points, m, gamma, num_clusters)
            self.clusters = new_clusters
        plt.figure(1)
        self.plot_data(points, self.clusters)

    def getInv(self, k):
        #calculate the inverse of a matrix with a little adjustment so that there is a greater chance that the matrix k will have an inverse
        m = 10 ** -8
        a = k + + np.eye(k.shape[1]) * m
        KInv = linalg.inv(k + + np.eye(k.shape[1]) * m)
        return KInv

    def compute_gamma(self, points, n, num_clusters):
        gamma = self.init_gamma(n, number_clusters)
        for i in range(num_clusters):
            dd = points - self.clusters[i].centroids
            dd = np.sum(np.dot(dd, self.getInv(self.clusters[i].covariance_matrices)) * dd, axis=1)
            gamma[i] = self.clusters[i].cluster_probability * np.exp(-dd / 2) / sqrt(np.linalg.det(self.clusters[i].covariance_matrices)) + 1e-11
        gamma = np.matrix(gamma)
        gamma = gamma / np.sum(gamma, axis=0)
        return gamma

    def compute_clusters(self, points, m, gamma, num_clusters):
        #In this method we will update the clusters based on the previous ones
        new_clusters = []
        for i in range(num_clusters):
            new_cluster = Cluster()
            new_cluster.zeroValues(m, num_clusters)
            new_clusters.append(new_cluster)
        total_probability = np.sum(gamma)

        for i in range(num_clusters):
            new_clusters[i].centroids = np.sum(np.multiply(points, gamma[i].T), axis=0) / np.sum(gamma[i])
            new_clusters[i].centroids = np.asarray(new_clusters[i].centroids).ravel()
            tmp = points - new_clusters[i].centroids
            new_clusters[i].covariance_matrices =np.array((np.dot(tmp.T, np.multiply(tmp, gamma[i].T)) / np.sum(gamma[i])).T)
            new_clusters[i].cluster_probability = np.sum(gamma[i]) / total_probability
        return new_clusters

    def KL(self, centroid1, covariance1, centroid2, covariance2):
            #Kullback-Leibler divergence value for two clusters.
        dm = centroid1 - centroid2
        comp1 = np.log(np.linalg.det(np.dot(covariance2, np.linalg.inv(covariance1))))
        comp2 = np.trace(np.dot((covariance2 - covariance1), np.linalg.inv(covariance2)))
        comp3 = np.dot((np.dot(dm, np.linalg.inv(covariance2))), dm.T)
        divergence = (comp1 - comp2 + comp3) / 2
        return divergence

    def calculate_clusters_closeness(self, num_clusters):
        #calculate the most appropriate clusters based on the KL method
        jammed_clusters = [
            [self.KL(self.clusters[i].centroids,
                self.clusters[i].covariance_matrices,
                self.clusters[j].centroids,
                self.clusters[j].covariance_matrices,
                ), i, j]
            for i in range(num_clusters)
            for j in range(i + 1, num_clusters)
        ]

        jammed_clusters.sort(key=lambda x: x[0])

        return jammed_clusters

    def calculate_loosest_cluster(self, points, num_clusters, clusters):
        #compute the cluster that is the lossest compared with its points
        new_points_class = [[] for _ in range(num_clusters)]
        mean_distance_cluster = [0 for _ in range(num_clusters)]
        for point in points:
            smallest_distance = 100000
            centroid_Id = 0
            Id = 0
            for el in clusters:
                a = np.array(point)
                b = el.centroids
                distance = np.linalg.norm(a - b)
                if distance < smallest_distance:
                    centroid_Id = Id
                    smallest_distance = distance

                Id += 1
            mean_distance_cluster[centroid_Id] += smallest_distance
            new_points_class[centroid_Id].append([point[0], point[1]])

        biggest_id = -1
        biggest_mean = 0
        for i in range(num_clusters):
            mean_distance_cluster[i] /= len(new_points_class[i])
            if mean_distance_cluster[i] > biggest_mean:
                biggest_mean = mean_distance_cluster[i]
                biggest_id = i

        return biggest_id

    def move_clusters(self, jammed_cluster_index, loose_cluster_index, m):
        #move the more jammed cluster next to the more loosen one
        new_clusters = deepcopy(self.clusters)
        new_clusters[jammed_cluster_index].centroids = new_clusters[loose_cluster_index].centroids
        new_clusters[jammed_cluster_index].covariance_matrices= np.diag([1 for _ in range(m)])
        return new_clusters

    def improved_em(self, iterations, points, n, m, num_clusters, maxRuns=1000):
        self.em(iterations, points, n, m, num_clusters, False)
        stillJammed = True
        previous = [-1, -1]

        i = 0
        while stillJammed or maxRuns:

            closest_clusters = self.calculate_clusters_closeness(num_clusters)

            jammed_cluster_index = closest_clusters[0][1]
            loose_cluster_index = self.calculate_loosest_cluster(points, num_clusters, self.clusters)

            if [loose_cluster_index, jammed_cluster_index] == previous:
                break

            self.clusters =  self.move_clusters(jammed_cluster_index, loose_cluster_index, m)

            self.em(iterations, points, n, m, num_clusters, True)

            print('Run: ' + str(i))
            print('jammed id: {} | loose id: {}'.format(jammed_cluster_index, loose_cluster_index))
            previous = [loose_cluster_index, jammed_cluster_index]
            i += 1
            maxRuns -= 1





dim_data = 2
em = EmAlg(dim_data)
points, n, m = em.read_data('data_input.txt')
number_clusters = 5
em.improved_em(100, points, n, m, number_clusters)