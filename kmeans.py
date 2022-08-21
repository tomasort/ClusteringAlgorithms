from cluster import Cluster


class KMeans:
    def __init__(self, centroids, dist_func=lambda a, b: sum((a_ - b_) ** 2 for a_, b_ in zip(a, b))):
        self.centroids = centroids
        self.distance_func = dist_func
        self.clusters = [Cluster(center=c) for c in self.centroids]

    def train(self, training_data):
        previous_clusters = [None for _ in self.clusters]
        while not self.clusters == previous_clusters:
            previous_clusters = self.clusters.copy()
            self.clusters = [Cluster(center=c.center)
                             for c in previous_clusters]
            # Assign all the points to the nearest centroid
            for data_point in training_data:
                # Find the closest centroid
                closest_cluster_indx, closest_cluster_distance = None, None
                for i, cluster in enumerate(self.clusters):
                    d = self.distance_func(cluster.center, data_point[:-1])
                    if closest_cluster_indx is None or d < closest_cluster_distance:
                        closest_cluster_indx, closest_cluster_distance = i, d
                self.clusters[closest_cluster_indx].add(
                    data_point, update_center=False)
            for c in self.clusters:
                c.update_center()

    def test(self, testing_data):
        clusters = [Cluster(center=c.center) for c in self.clusters]
        for data_point in testing_data:
            # Find the closest centroid
            closest_cluster_indx, closest_cluster_distance = None, None
            for i, cluster in enumerate(self.clusters):
                d = self.distance_func(cluster.center, data_point[:-1])
                if closest_cluster_indx is None or d < closest_cluster_distance:
                    closest_cluster_indx, closest_cluster_distance = i, d
            clusters[closest_cluster_indx].add(data_point, update_center=False)
        return clusters
