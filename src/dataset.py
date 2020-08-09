import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
import glob
import av
import torchvision
import os
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
import copy
import pickle
from sklearn.cluster import KMeans
import cv2
from .kmeans import KMeansCluster
import logging
from PIL import Image
from sklearn.utils import shuffle

logger = logging.getLogger("Dataloader")


class BaseDataset(Dataset):
    def __init__(self,transforms=None):
        self.transforms = transforms

    def get_transformed_data(self,data):
        if isinstance(self.transforms,list):
            transformed_data = []
            for transform in self.transforms:
                transformed_data.append(transform(data))
        else:
            transformed_data = self.transforms(data)

        return transformed_data


class YoutubeDataset(BaseDataset):
    ALLOWDED_EXTS = [".mov",".mp4",".m4a",".3gp",".3g2",".mj2"]
    def __init__(self, basepath, batch=4, transforms=None, skip_initial=5, fps=6,
                 batch_per_video=None, reference_frames=3, max_batchs=None,shuffle=True,categories=None, **kwargs):
        super(YoutubeDataset, self).__init__(transforms=transforms)
        num_samples = reference_frames + 1
        self.basepath = basepath
        assert batch % num_samples == 0, f"Batch size must be multiple of {num_samples}"
        self.batch = batch
        self.num_samples = num_samples
        self.max_batchs = max_batchs
        self.count_total_batch = 0
        self.batch_per_video = batch_per_video
        self.skip_initial = skip_initial
        self.out_fps = fps
        self.shuffle = shuffle
#         self.categories = f"*[{','.join(categories)}]"  if categories else "*"
        self.categories = categories if categories else ["*"]
        video_paths = []
        self.video_paths =  []
        for cat in self.categories:
            self.video_paths.extend(glob.glob(os.path.join(self.basepath, cat, f"*[{','.join(YoutubeDataset.ALLOWDED_EXTS)}]")))

    def __len__(self):
    
        if self.max_batchs:
            l = self.max_batchs
        elif self.batch_per_video:
            l = len(self.video_paths) * self.batch_per_video
        else:
            l = len(self.video_paths)
            
        print(f"data size : {l}")
        return l

    def __getitem__(self, idx):
        self.count_total_batch = 0
        video_path = self.video_paths[idx]
        for frames in self._read_frames_cv(video_path):
            if self.max_batchs and self.max_batchs == self.count_total_batch:
                break
            yield frames
        logger.debug(f"Total batches yielded {self.count_total_batch} of size {self.batch}")

    def __iter__(self):
        if self.shuffle:
            video_paths = shuffle(self.video_paths)
        else:
            video_paths = self.video_paths
        self.count_total_batch = 0
        batch = []
        for path in video_paths:
            # break when reached max batch
            if self.max_batchs and self.max_batchs == self.count_total_batch:
                break

            logger.debug(path)
            for batch in self._read_frames(path, batch):
                if self.max_batchs and self.max_batchs == self.count_total_batch:
                    break
                if batch:
                    self.count_total_batch += 1
                    logger.debug(f"total batch {self.count_total_batch}")
                    yield batch
                else:
                    print("failed")

        logger.debug(f"Total batches yielded {self.count_total_batch} of size {self.batch}")

    def _read_frames(self, path, image_batch):
        video_container = av.open(path)
        video_stream = video_container.streams.video[0]
        video_fps = video_stream.average_rate
        video_fps = int(video_fps.numerator / video_fps.denominator)

        offset = int(self.skip_initial / video_stream.time_base / video_fps)
        logger.debug(f"offset {offset}")
        video_container.seek(offset, any_frame=False, stream=video_stream)

        frame_num = 0
        count_batch = 0
        frames = []
        samples = []
        
        skip = video_fps // self.out_fps

        logger.debug(f"video fps: {video_fps}, skipping {skip}")
        
        for packet in video_container.demux():
            if self.batch_per_video and (len(frames)/self.batch)==self.batch_per_video:
                logger.debug(f"video batch limit reached: {len(frames)/self.batch}/{self.batch_per_video}")
                break

            if packet.stream.type == 'video':
                for frame in packet.decode():
                    frame_num += 1
                    if frame_num % (skip + 1) == 0:
                        image_array = frame.to_image()  # to_rgb().to_ndarray()
                        frames.append(image_array)
        video_container.close()
        
        for image_array in frames:
            samples.append(image_array)
            if len(samples) == self.num_samples:
                logger.debug(f"num samples readed {len(samples)}")
                image_batch.extend(samples)
                samples = []

            if len(image_batch) == self.batch:
                logger.debug(f"batch size readed {len(image_batch)}")
                return_batch = copy.deepcopy(image_batch)

                image_batch = []
                if self.transforms:
                    return_batch = self.get_transformed_data(return_batch)

                count_batch += 1
                logger.debug(f"video batch {count_batch}")
                yield return_batch
            
                        
    def _read_frames_cv(self, path, image_batch):
        logger.debug(f"opening video {path}")
        video_container = cv2.VideoCapture(path)
        video_fps = int(video_container.get(cv2.CAP_PROP_FPS))

#         offset = int(self.skip_initial / video_stream.time_base / video_fps)
#         logger.debug("offset {offset}")
#         video_container.seek(offset, any_frame=True, stream=video_stream)
#         video_container.set(cv2.CAP_PROP_POS_FRAMES,self.skip_initial)
        ok = video_container.isOpened()
        if ok:
            logger.debug("video opened successfully")
        else:
            logger.debug("video opened failed")
        frame_num = 0
        count_batch = 0
        
        skip = video_fps // self.out_fps

        logger.debug(f"video fps: {video_fps}, skipping {skip}")

        samples = []

        while ok:
            if self.batch_per_video and self.batch_per_video == count_batch:
                video_container.release()
                break
            frame_num += 1
            if frame_num % (skip + 1) == 0:
                ok,image_array = video_container.read() 
                if not ok:
                    video_container.release()
                    break
                image_array = Image.fromarray(image_array)
                samples.append(image_array)
            else:
                _,_ = video_container.read()
                continue

            if len(samples) == self.num_samples:
                logger.debug(f"num samples readed {len(samples)}")
                image_batch.extend(samples)
                samples = []

            if len(image_batch) == self.batch:
                logger.debug(f"batch size readed {len(image_batch)}")
                return_batch = copy.deepcopy(image_batch)

                image_batch = []
                if self.transforms:
                    return_batch = self.get_transformed_data(return_batch)

                count_batch += 1
                self.count_total_batch += 1
                logger.debug(f"video batch {count_batch}")
                logger.debug(f"total batch {self.count_total_batch}")
                yield return_batch


class DavisDataset(BaseDataset):
    def __init__(self, basepath, batch=4, transforms=None, skip_initial=5, fps=6,
                 batch_per_video=None, max_batchs=None, reference_frames=3, shuffle=True, **kwargs):
        super(DavisDataset, self).__init__(transforms=transforms)
        self.basepath = basepath
        num_samples = reference_frames + 1
        assert batch % num_samples == 0, f"Batch size ({batch}) must be multiple of num_samples ({num_samples})"
        self.shuffle = shuffle
        self.batch = batch
        self.num_samples = num_samples
        self.max_batchs = max_batchs
        self.count_total_batch = 0
        self.batch_per_video = batch_per_video
        self.skip_initial = skip_initial
        self.out_fps = fps
        self.count_images = 0
        self.transforms = transforms
        self.video_paths = glob.glob(os.path.join(self.basepath, "*"))

#     def __len__(self):
#         if self.max_batchs:
#             l = self.max_batchs
#         else:
#             images = glob.glob(os.path.join(self.basepath, "*", "*"))
#             l = (len(images) // self.batch)

#         return l

    def _read_images(self, path, batch):
        _samples = []
        _count_samples = 0
        images_path = glob.glob(os.path.join(path, "*"))
        images_path = sorted(images_path,key=lambda x: int(os.path.basename(x).split(".")[0]))
        logger.debug(f"images found : {len(images_path)}")
        for image_path in images_path:

            # break when reach size (`batchs_per_video`) for a particular video sequence
            if self.batch_per_video and self.batch_per_video == _count_samples or self.max_batchs and self.max_batchs == self.count_total_batch:
                break

            img = Image.open(image_path)
#             print(image_path)
            self.count_images += 1

            _samples.append(img)

            if len(_samples) == self.num_samples:
                batch.extend(_samples)
                _samples = []
                _count_samples += 1

            if len(batch) == self.batch:
                self.count_total_batch += 1
                logger.debug(f"batch size {len(batch)}")
                logger.debug(f"count batches {self.count_total_batch}")
                logger.debug(f"count images {self.count_images}")
                if self.transforms:
                    batch = self.get_transformed_data(batch)
                yield batch
                batch = []

        # if batch not upto size return None
        return None

    def __iter__(self):
        if self.shuffle:
            video_paths = shuffle(self.video_paths)
        else:
            video_paths = self.video_paths

        self.count_total_batch = 0
        batch = []
        for video_sequence_path in video_paths:
            logger.debug(video_sequence_path)
            # break when reachs max batch size specified or read complete directory
            if self.max_batchs and self.max_batchs == self.count_total_batch:
                break

            for batch in self._read_images(video_sequence_path, batch):
                if batch is None:
                    continue
                # copy the batch and clear if
                images = copy.deepcopy(batch)
                batch = []
                yield images


class KineticsDataset:
    pass


def get_data_loader(name, data_root, batch_size, shuffle=True,transforms=None, batch_per_video=None,
                   reference_frames=3, max_batches=None, categories=None, **kwargs):
    datasets = {"davis":DavisDataset,
                "youtube":YoutubeDataset,
                "kinetics":KineticsDataset}
    dataset_instance = datasets.get(name)
    dataset = dataset_instance(basepath=data_root, batch=batch_size, transforms=transforms,
                               batch_per_video=batch_per_video, max_batchs=max_batches, shuffle=shuffle,
                               reference_frames=reference_frames,categories=categories, **kwargs
                               )
    if not dataset:
        raise KeyError(f"Available datasets {datasets.keys()}")

    return dataset
