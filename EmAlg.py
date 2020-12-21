from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
from numpy.linalg import linalg

class Cluster():

    def initializeValues(self, clusters_number, mtx_cov, data_dim):
        self.centroids = ((2*np.random.random((1,data_dim)) - 1)*15)[0]
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
        print('{} contains: {} lines, with {} {}s per line.'
              .format(file, rows, cols, type(points[0][0]).__name__))

        return np.array(points), rows, cols


    def plot_data(self, points, cent_clusters):
        colors = iter(['red', 'yellow', 'pink', 'green', 'purple', 'black'])
        plt.scatter(points[:,0], points[:,1], s=2.5)
        if(cent_clusters):
            for c in cent_clusters:
                plt.scatter(c.centroids[0], c.centroids[1], color=next(colors), s=200)
        plt.show()

    def initialize_clusters(self, points, num_clusters, m):
        for i in range(num_clusters):
            cluster = Cluster()
            cluster.initializeValues(num_clusters, m,self.dim_data)
            self.clusters.append(cluster)
        self.plot_data(points, self.clusters)

    def init_weight(self, n, k):
        row = [0] * n
        w = [np.array(row)] * k
        return w

    def em(self, iterations, points, n, m, num_clusters, isImproved):
        if not isImproved:
            self.initialize_clusters(points, num_clusters, m)
        for i in range(iterations +1):
            last_guess = self.new_weight_approx(points, n, num_clusters)
            new_clusters = self.compute_clusters(points, m, last_guess, num_clusters)
            self.clusters = new_clusters

        plt.figure(1)
        self.plot_data(points, self.clusters)
        return last_guess

    def getInv(self, k):
        m = 10 ^ -20
        KInv = linalg.inv(k + + np.eye(k.shape[1]) * m)
        return KInv

    def new_weight_approx(self, points, n, num_clusters):
        weight = self.init_weight(n, number_clusters)
        for i in range(num_clusters):
            print(self.clusters[i])
            z = points - self.clusters[i].centroids
            print(z)
            print(self.clusters[i].covariance_matrices)
            d = np.sum(np.dot(z, self.getInv(self.clusters[i].covariance_matrices)) * z, axis=1)
            weight[i] = self.clusters[i].cluster_probability * np.exp(-d / 2) / \
                        sqrt(np.linalg.det(2 * pi * self.clusters[i].covariance_matrices))
        weight = np.matrix(weight)
        weight = weight / np.sum(weight, axis=0)
        return weight

    def compute_clusters(self, points, m, last, num_clusters):
        new_clusters = []
        for i in range(num_clusters):
            new_cluster = Cluster()
            new_cluster.zeroValues(m, num_clusters)
            new_clusters.append(new_cluster)
        total_probability = np.sum(last)

        for i in range(num_clusters):
            new_clusters[i].centroids = np.sum(np.multiply(points, last[i].T), axis=0) / np.sum(last[i])
            new_clusters[i].centroids = np.asarray(new_clusters[i].centroids).ravel()
            tmp = points - new_clusters[i].centroids
            new_clusters[i].covariance_matrices =np.array((np.dot(tmp.T, np.multiply(tmp, last[i].T)) / np.sum(last[i])).T)
            new_clusters[i].cluster_probability = np.sum(last[i]) / total_probability
        return new_clusters

    def KL(self, centroid1, cov1, centroid2, cov2):
        """
            Kullback-Leibler divergence value for two clusters.
        """
        dm = centroid1 - centroid2
        comp1 = np.log(np.linalg.det(np.dot(cov2, np.linalg.inv(cov1))))
        comp2 = np.trace(np.dot((cov2 - cov1), np.linalg.inv(cov2)))
        comp3 = np.dot((np.dot(dm, np.linalg.inv(cov2))), dm.T)
        a = np.asarray((comp1 - comp2 + comp3) / 2)
        divergence = (comp1 - comp2 + comp3) / 2
        return divergence

    def calculate_clusters_closeness(self, num_clusters):
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

    def calculate_cluster_looseness(self, clusters_meta, num_clusters):
        #todo modify the method, make you own
        print(num_clusters)
        cls_id = 0
        maxim = clusters_meta[0]['std_dev']
        for i in range(1, num_clusters):
            if clusters_meta[i]['std_dev'] > maxim:
                maxim = clusters_meta[i]['std_dev']
                cls_id = i
        return cls_id

    def move_clusters(self, jammed_cluster_index, loose_cluster_index, m):
        new_params = deepcopy(self.clusters)
        new_params[jammed_cluster_index].centroids = new_params[loose_cluster_index].centroids
        new_params[jammed_cluster_index].covariance_matrices= np.diag([1 for _ in range(m)])
        return new_params

    def measures_condensed_nature_of_clusters(self, last_guess, points, num_clusters, params):
        """
        Storing kind of a meta-data for each cluster, structred as a dict with:
            data - list of points (3D/2D and so on)
            std_dev - standard deviation from the centroid
        """

        clusters_meta = {}
        for i in range(num_clusters):
            clusters_meta[i] = {'data': np.empty(shape=(600, 2)), 'std_dev': -1}

        i = 0
        for cls_id in np.argmax(last_guess.T, axis=1):
            cls_id = np.asarray(cls_id)[0][0]
            np.append(clusters_meta[cls_id]['data'],points[i])
            i += 1

        for i in range(num_clusters):
            a = clusters_meta[i]['data']
            b = params[i].centroids
            clusters_meta[i]['std_dev'] = np.mean(
                np.linalg.norm(clusters_meta[i]['data'] - params[i].centroids.tolist()))

        return clusters_meta

    def improve_em(self, iterations, points, n, m, num_clusters):
        last_guess = self.em(iterations, points, n, m, num_clusters, False)

        jammed = True
        previous = [-1, -1]

        i = 0
        while jammed:

            # (i) Measures the pairwise distence between cluster centres and selects the closest pair
            jammed_clstrs = self.calculate_clusters_closeness(num_clusters)

            if len(jammed_clstrs) == 0:
                break

            jammed_cluster = jammed_clstrs[0][1]

            # (ii) Measures the condensed nature of clusters and the "loosest" one is selected
            clusters_meta = self.measures_condensed_nature_of_clusters(last_guess, points, num_clusters, self.clusters)
            loose_cluster = self.calculate_cluster_looseness(clusters_meta, num_clusters)

            if [loose_cluster, jammed_cluster] == previous:
                break

            # # (iii) One of the centres from (i) is moved to the vicinity of the centre of (ii)
            self.clusters = self.move_clusters(jammed_cluster, loose_cluster, m)

            # Give them (params) another spin
            last_guess = self.em(iterations, points, n, m, num_clusters, True)

            print('### epoch-{}: '.format(i))
            print('jammed id: {} | loose id: {}'.format(jammed_cluster, loose_cluster))
            print('norm: {}'.format(
                [[clusters_meta[i]['std_dev'], len(clusters_meta[i]['data'])] for i in range(num_clusters)]))
            previous = [loose_cluster, jammed_cluster]
            i += 1

        return self.clusters, last_guess


dim_data = 2
em = EmAlg(dim_data)
points, n, m = em.read_data('data_input.txt')
number_clusters = 5
# em.em(5,points, n, m, number_clusters, False)
em.improve_em(5, points, n, m, number_clusters)
# cent_clusters, cov_clusters, pr_clusters = init_mixture_parameters(train_data,n_clus)
# em.plot_data(points, cent_clusters = [])