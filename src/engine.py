import os
import pickle

import numpy as np

from . import net
from .dataset import get_data_loader
from .kmeans import KMeansCluster
from .config import Config
from torch import nn
import logging
import tqdm
import torch
from .transforms import OneHotEncoding, ConvertChannel, QuantizeAB, ToNumpy
from torchvision.transforms import ToTensor, Resize, Grayscale, Compose, Lambda, Normalize
from sklearn.utils import shuffle
from PIL import Image
from matplotlib import pyplot as plt
import cv2

config = Config.get_config()
logger = logging.getLogger(__name__)
config.setup_logger(logger)

if config.WANDB:
    try:
        import wandb
    except ModuleNotFoundError:
        logger.error(f"wandb not found install wandb and add api keys to env and rerun, turning sync off for now")
        config.WANDB = False


class Engine:
    def __init__(self, mode):
        self.logger = logging.getLogger(f"{mode}_engine")
        config.setup_logger(self.logger)

        # model
        self.mode = mode
        self.model = net.get_colorization_network(config.BACKBONE,config.HEAD_NETWORK_VERSION)
        self.model.to(config.DEVICE)
        self.initial_epoch = 1
        if config.RESUME:
            if os.path.isfile(config.MODEL_PATH):
                checkpoint = torch.load(config.MODEL_PATH)
                self.model.load_state_dict(checkpoint["state_dict"])
                self.initial_epoch = int(checkpoint["epoch"])+1
                print(f"loaded model {config.MODEL_PATH}, epochs {self.initial_epoch}")
            else:
                self.logger.error(f"Error loading file not found {config.MODEL_PATH}")

        # loss
        self.criterion = Engine.get_loss(config.LOSS)

        # data
        self.data_root = None

        # wandb
        if config.WANDB:
            exp_id = ''
            for ch in config.EXPERIMENT_NAME:
                exp_id += ch if ch.isalnum() else ''
            self.logger.info(f"wandb experiment ID {exp_id}")
            wandb.init(project=config.PROJECT_NAME, name=config.EXPERIMENT_NAME,
                       config=config.get_config_dict(config), resume=config.RESUME, id=exp_id,
                       dir=config.WANDB_LOG_ROOT)
            wandb.watch(self.model)
            config.MODEL_LOGGING = wandb.run.dir

    def __repr__(self):
        return f"engine:{config.PROJECT_NAME}:{config.EXPERIMENT_NAME}"

    def run(self):
        raise NotImplementedError("Implement in child class")

    @staticmethod
    def get_optimizer(model, name):
        optimizers = {
            "Adam": {"instance": torch.optim.Adam,
                     "params": {"lr": config.BASE_LR}},

            "SGD": {"instance": torch.optim.SGD,
                    "params": {"lr": config.BASE_LR,
                               "momentum": config.LEARNING_MOMENTUM}}}
        
        optimizer = optimizers.get(name, None)
        if optimizer:
            my_optimizer = optimizer["instance"](model.parameters(), **optimizer["params"])
        else:
            raise KeyError(f"Available optimizers {optimizer.keys()}")
        return my_optimizer

    @staticmethod
    def get_loss(name):
        losses = {"cross_entropy": nn.CrossEntropyLoss}
        my_loss = losses.get(name, None)
        if not my_loss:
            raise KeyError(f"Available loss functions {losses.keys()}")
        return my_loss()

    @staticmethod
    def get_learning_scheduler(name, optimizer, **kwargs):
        print("warning hard coded lambda")
        schedulers = {"LambdaLR": torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                                    lr_lambda=lambda epoch: 1 if epoch < 1000 else 0.1)}
        my_scheduler = schedulers.get(name, None)
        if not my_scheduler:
            raise KeyError(f"{name} not found, Available schedulers {schedulers.keys()}")
        return my_scheduler

    @staticmethod
    def get_data_loader(data_config, transforms=None):
        data_loader = get_data_loader(data_config.DATASET, data_root=data_config.DATA_ROOT,
                                      batch_size=data_config.BATCH_SIZE,
                                      shuffle=data_config.SHUFFLE, transforms=transforms,
                                      batch_per_video=data_config.BATCH_PER_VIDEO,
                                      reference_frames=data_config.REFERENCE_FRAMES,
                                      max_batches=data_config.MAX_BATCHES,
                                      fps=data_config.VIDEO_FPS, skip_initial=data_config.SKIP_INITIAL,categories=data_config.CATEGORIES)
        return data_loader

    def visualize(self):
        pass


class TrainEngine(Engine):
    def __init__(self):
        super(TrainEngine, self).__init__(mode="train")
        # model
        self.model.train()
        
        # colorisation
        self.kmeans = None
        self.kmeans_refit = config.KMEANS_REFIT

        if self.kmeans_refit:
            self.kmeans = self.color_clustering()

        elif os.path.isfile(config.KMEANS_FILE):
            from .kmeans import KMeansCluster
            try:
                with open(config.KMEANS_FILE, "rb") as f:
                    self.kmeans = pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error{e} Loading Kmeans file {config.KMEANS_FILE}")
                self.kmeans = self.color_clustering()
        else:
            raise Exception(f"Error in getting kmeans clustering {config.KMEANS_FILE}")

        # dataset
#         transforms = self.get_transforms(self.kmeans)
        transforms = None
        if not os.path.isdir(config.TrainData.DATA_ROOT):
            raise NotADirectoryError(f"data directory {config.TrainData.DATA_ROOT} does not exist")
        self.train_data_loader = self.get_data_loader(config.TrainData, transforms)

        if not os.path.isdir(config.TestData.DATA_ROOT):
            raise NotADirectoryError(f"data directory {config.TestData.DATA_ROOT} does not exist")
        self.test_data_loader = self.get_data_loader(config.TestData, transforms)

        # training
        self.optimizer = Engine.get_optimizer(self.model, config.OPTIMIZER)
        self.scheduler = None
        if config.LEARNING_SCHEDULER:
            self.scheduler = Engine.get_learning_scheduler(config.LEARNING_SCHEDULER, self.optimizer,
                                                           lr_lambda=config.LR_LAMBDA)

    def color_clustering(self):
        transforms = Compose([ConvertChannel()])
        self.train_data_loader = self.get_data_loader(config.KMeansData, transforms)
        samples = np.array(None)
        for batch in self.train_data_loader:
            image_array = batch[0, :, :, 1:].reshape(-1, 2) / 255
            image_array = shuffle(image_array, random_state=0)[:5000]
            if samples.all() is None:
                samples = image_array
            else:
                samples = np.row_stack([samples, image_array])

        kmeans = KMeansCluster(n_clusters=config.CLUSTERS)
        kmeans.fit(samples, sub_samples=config.KMEANS_SAMPLES)

        with open(config.KMEANS_FILE, "wb") as f:
            pickle.dump(kmeans, f)

        return kmeans

    @staticmethod
    def get_transforms(kmeans):
        clusters = kmeans.kmeans.n_clusters
        _transform_colorisation = Compose([Resize((32, 32)), ToNumpy(), ConvertChannel(),
                                           QuantizeAB(kmeans), OneHotEncoding(clusters),
                                           ToTensor()])
        transform_colorisation = Compose(
            [Lambda(lambda batch: torch.stack([_transform_colorisation(im) for im in batch]))])

        _transform_training = Compose(
            [Resize((256, 256)), Grayscale(), ToTensor(), Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)])
        transform_training = Compose(
            [Lambda(lambda batch: torch.stack([_transform_training(im) for im in batch]))])

        _transform_testing = Compose([Resize((256, 256)), ToNumpy(), ConvertChannel()])
        transform_testing = Compose([Lambda(lambda batch: [_transform_testing(im) for im in batch])])

        return [transform_training, transform_colorisation, transform_testing]

    def __repr__(self):
        return f"train:{config.PROJECT_NAME}:{config.EXPERIMENT_NAME}"

    def reconstruct_image(self, quantized_image, orignal_image):
        if isinstance(quantized_image, torch.Tensor):
            quantized_image = quantized_image.to("cpu").numpy()

        reconstructed_ab = (self.kmeans.recreate_image(quantized_image.flatten(), 32, 32) * 255).astype("uint8")
        orignal_image = Image.fromarray(orignal_image, mode="LAB")

        reduced_orignal = orignal_image.copy()
        reduced_orignal_l = np.array(reduced_orignal.resize((32, 32)))[:, :, 0]

        # print("reduced_orignal_l l",reduced_orignal_l.shape)
        reconstructed = np.concatenate([np.expand_dims(reduced_orignal_l, axis=-1), reconstructed_ab]
                                       , axis=-1)

        reconstructed = Image.fromarray(reconstructed, mode="LAB")

        upsampled = np.array(reconstructed.resize((256, 256), 0))
        upsampled_ab = upsampled[:, :, 1:]

        orignal_l = np.array(orignal_image)[:, :, 0].copy()
        upsampled_reconstructed = np.concatenate([np.expand_dims(orignal_l, axis=-1), upsampled_ab], axis=-1)

        return [cv2.cvtColor(np.array(orignal_image).astype("uint8"), cv2.COLOR_Lab2RGB),
                cv2.cvtColor(np.array(reconstructed).astype("uint8"), cv2.COLOR_Lab2RGB),
                cv2.cvtColor(upsampled_reconstructed.astype("uint8"), cv2.COLOR_Lab2RGB)]

    def show(self, image, i=0, cmap=None):
        plt.figure(num=i)
        plt.axis("off")
        plt.imshow(image, cmap=cmap)

    def run(self):
        transforms = self.get_transforms(self.kmeans)
        step = 0
        
        for epoch in range(self.initial_epoch, config.EPOCHS):
            data_loader = tqdm.tqdm(self.train_data_loader)

            logs = dict()
            epoch_loss = 0
            batch_index = 0
            visual_count = 0
            examples = []

            for batch_index, raw_data in enumerate(data_loader):
                data = []
                for transform in transforms:
                    data.append(transform(raw_data))
                step += 1
                training_batch, labels_batch, testing_batch = data
                training_batch = training_batch.to(config.DEVICE)
                labels_batch = labels_batch.to(config.DEVICE)
                similarity_matrix = self.model(training_batch, config.TrainData.REFERENCE_FRAMES)
                predicted_colors, true_colors = self.model.proxy_task(similarity_matrix, labels_batch,
                                                                      num_ref=config.TrainData.REFERENCE_FRAMES)
                # print(predicted_colors.shape,true_colors.shape)
                predicted = predicted_colors.reshape(-1, 16)
                target = torch.argmax(true_colors.reshape(-1, 16), dim=-1)
                loss = self.criterion(predicted, target)
                epoch_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                data_loader.set_description(f"  loss:{loss} ")
                if epoch % config.VISUALIZATION_EPOCH == 0 and visual_count <= config.VISUALIZATION_COUNT:
                    predicted_colors = torch.argmax(predicted_colors[0].permute([1,2,0]), dim=-1)
                    # print(true_colors.shape)
                    true_colors = torch.argmax(true_colors[0].permute([1,2,0]), dim=-1)
                    
                    self.logger.debug(f"reconstruct image predicted {predicted_colors.shape}, true {true_colors.shape}")

                    orignal_image = testing_batch[config.TrainData.REFERENCE_FRAMES]
                    orignal_image, reconstructed, upsampled_reconstructed = self.reconstruct_image(predicted_colors,
                                                                                                   orignal_image)
                    refrence_image = testing_batch[config.TrainData.REFERENCE_FRAMES-1]
                    refrence_image = cv2.cvtColor(np.array(refrence_image).astype("uint8"), cv2.COLOR_Lab2RGB)
                    # self.show(orignal_image,0)
                    # self.show(reconstructed,1)
                    # self.show(upsampled_reconstructed,2)
                    # self.show(predicted_colors.reshape(32,32).cpu().numpy(),3,cmap="tab20")
                    # self.show(true_colors.reshape(32,32).cpu().numpy(),4,cmap="tab20")
                    # for i,img in enumerate(testing_batch):
                    #     img = cv2.cvtColor(img.astype("uint8"),cv2.COLOR_Lab2RGB)
                    #     self.show(img,5+i)

                    if config.WANDB:
                        examples.extend([wandb.Image(refrence_image,caption="Refrence frame"),
                                         wandb.Image(orignal_image, caption=f"orignal image {batch_index}"),
                                         wandb.Image(upsampled_reconstructed,
                                                     caption=f"predicted image {batch_index}")])
                        logs["training visualization"] = examples


                    visual_count += 1
            data_loader.total = batch_index
            print(f"epoch {epoch}, loss: {epoch_loss / (batch_index + 1)}")
            logs.update({"epoch": epoch,
                         "training loss": epoch_loss / (batch_index + 1)
                         })

            if config.WANDB:
                #                 print("logging",step)
                wandb.log(logs, step=epoch)
                logs = {}
                examples = []

            if self.scheduler:
                self.scheduler.step()
                logs["lr"] = self.scheduler.get_last_lr()[0]

            if epoch % config.SAVE_EPOCH == 0:
                torch.save({"state_dict": self.model.state_dict(), "epoch": epoch},
                           os.path.join(config.MODEL_LOGGING, f'model_{epoch}.pt'))
