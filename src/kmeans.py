from sklearn.cluster import KMeans
import numpy as np
import pickle
from torchvision import transforms
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from time import time
import logging

logger = logging.getLogger("KMeans")


class KMeansCluster:
    def __init__(self, n_clusters=16):
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit(self, image_array, sub_samples=1000):
        logger.debug("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:sub_samples]
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(image_array_sample)
        logger.debug("done in %0.3fs." % (time() - t0))

    def predict(self, image_array):
        # Get labels for all points
        logger.debug("Predicting color indices on the full image (k-means)")
        t0 = time()
        labels = self.kmeans.predict(image_array)
        logger.debug("done in %0.3fs." % (time() - t0))
        return labels

    def recreate_image(self, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        codebook = self.kmeans.cluster_centers_
        d = codebook.shape[1]
        image = np.zeros((h, w, d))
        label_idx = 0
        for i in range(h):
            for j in range(w):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    @staticmethod
    def display(r_image, fig_num=0, cmap=None, title=None):
        plt.figure(fig_num, figsize=(10, 15))
        plt.clf()
        plt.axis('off')
        plt.title(title)
        plt.imshow(r_image, cmap=cmap)
