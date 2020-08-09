from sklearn.preprocessing import LabelBinarizer
import logging
import pickle
from .kmeans import KMeansCluster
import cv2
import numpy as np

import torch

logger = logging.getLogger("Transforms")


class QuantizeAB:
    def __init__(self, kmeans):

        # load kmeans model
        if isinstance(kmeans, str):
            logger.debug(f"loading KMeans from pickle : {kmeans}")
            #             try:
            with open(kmeans, "rb") as f:
                self.kmeans = pickle.load(f)
                logger.debug("Loaded successfully")

        #             except:
        #                 Exception("Error reading pickle for kmeans, provide different path or fit another model")
        elif isinstance(kmeans, KMeansCluster):
            self.kmeans = kmeans

        else:
            Exception("Kmeans should be pickle file or instance of KMeansCluster")

    def __call__(self, sample):
        if sample.ndim == 4:
            batch, h, w, c = sample.shape
            # extract ab channel
            sample_ab = sample[:, :, :, 1:].reshape(-1, 2) / 255
            labels = self.kmeans.predict(sample_ab)
            labels = labels.reshape(batch, h, w)
            return labels
        elif sample.ndim == 3:
            h, w, c = sample.shape
            sample_ab = sample[:, :, 1:].reshape(-1, 2) / 255
            labels = self.kmeans.predict(sample_ab)
            labels = labels.reshape(h, w)
            return labels
        else:
            Exception("Dimension of inpput sample must be batch x c x h x w (4) or c x h x w (3)")


class ConvertChannel:
    def __init__(self, in_channel="rgb"):
        channels = ["rgb", "bgr"]
        self.in_channel = in_channel
        assert self.in_channel in channels, f"Channel must be in {channels}"

    def __call__(self, sample):
        logger.debug(sample.shape)
        if sample.ndim == 4:
            output_samples = []
            for im in sample:
                if self.in_channel == "rgb":
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2Lab)
                elif self.in_channel == "bgr":
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)

                output_samples.append(im)
            return np.stack(output_samples, axis=0)
        elif sample.ndim == 3:
            im = sample
            if self.in_channel == "rgb":
                im = cv2.cvtColor(im, cv2.COLOR_RGB2Lab)
            elif self.in_channel == "bgr":
                im = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)

            return im


class ToNumpy(object):
    def __init__(self, dtype=None):
        self.dtype= dtype

    def __call__(self, image):
        return np.array(image,dtype=self.dtype)


class OneHotEncoding:
    def __init__(self, classes, num_samples=4):
        self.encoder = LabelBinarizer()
        self.num_samples = num_samples
        self.num_class = 0
        if isinstance(classes, int):
            self.num_class = classes
            classes = np.arange(classes)
            self.encoder.fit(classes)
        else:
            self.num_class = len(classes)
            self.encoder.fit(classes)

    def __call__(self, labels):
        h, w = labels.shape
        labels = labels.reshape(-1, )
        one_hot_labels = self.encoder.transform(labels).reshape(-1, self.num_class, 1)
        one_hot_labels = np.transpose(one_hot_labels, [0, 2, 1]).reshape(h, w, self.num_class)
        logger.debug(f"Onehotencoder, output shape {one_hot_labels.shape}")
        return one_hot_labels
