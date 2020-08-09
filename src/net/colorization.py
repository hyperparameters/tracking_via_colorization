import torch
import torchvision
import logging
from torch import nn
from .backbone import backbone_factory


def get_logger(name):
    return logging.getLogger(name)


class Block3D(nn.Module):
    def __init__(self, padding=1, dilation=None):
        super(Block3D, self).__init__()
        self._logger = get_logger("Block3D")
        self.conv1 = nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(256)
        self.conv2 = nn.Conv3d(256, 256, kernel_size=(3, 1, 1), stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Network3D(nn.Module):
    def __init__(self):
        super(Network3D, self).__init__()
        self._logger = get_logger("Network3D")
        config = [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8), (1, 16, 16)]
        self.block = Block3D

        self.block1 = self._make_layer(config[0], config[0])
        self.block2 = self._make_layer(config[1], config[1])
        self.block3 = self._make_layer(config[2], config[2])
        self.block4 = self._make_layer(config[3], config[3])
        self.block5 = self._make_layer(config[4], config[4])

        self.conv1 = nn.Conv3d(256, 64, kernel_size=1, stride=1, padding=0)
        self._logger.info(f"Colorization net Initialized with config {config}")

    def _make_layer(self, padding, dilation):
        return self.block(padding, dilation)

    def forward(self, input):
        features = input.unsqueeze(2)
        out = self.block1(features)
        self._logger.debug(f"shape after block1 {out.shape}")
        out = self.block2(out)
        self._logger.debug(f"shape after block2 {out.shape}")
        out = self.block3(out)
        self._logger.debug(f"shape after block3 {out.shape}")
        out = self.block4(out)
        self._logger.debug(f"shape after block4 {out.shape}")
        out = self.block5(out)
        self._logger.debug(f"shape after block5 {out.shape}")
        out = self.conv1(out)
        out = torch.squeeze(out)
        return out


class Network3Dv2(nn.Module):
    def __init__(self):
        super(Network3Dv2, self).__init__()
        self._logger = get_logger("Network3D")
        config = [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8), (1, 16, 16)]
        self.block = Block3D

        self.block1 = self._make_layer(config[0], config[0])
        self.block2 = self._make_layer(config[1], config[1])
        self.block3 = self._make_layer(config[2], config[2])
        self.block4 = self._make_layer(config[3], config[3])
        self.block5 = self._make_layer(config[4], config[4])

        self.conv1 = nn.Conv3d(256, 64, kernel_size=1, stride=1, padding=0)
        self._logger.info(f"Colorization net Initialized with config {config}")

    def _make_layer(self, padding, dilation):
        return self.block(padding, dilation)

    def forward(self, input, num_ref=3):
        num_samples = num_ref + 1
        batch, c, h, w = input.shape
        self._logger.debug(f"input shape {input.shape}")
        features = input.reshape(-1, num_samples, c, h, w).permute([0, 2, 1, 3, 4])
        self._logger.debug(f"input shape after reshape and permute {features.shape}")
        out = self.block1(features)
        self._logger.debug(f"shape after block1 {out.shape}")
        out = self.block2(out)
        self._logger.debug(f"shape after block2 {out.shape}")
        out = self.block3(out)
        self._logger.debug(f"shape after block3 {out.shape}")
        out = self.block4(out)
        self._logger.debug(f"shape after block4 {out.shape}")
        out = self.block5(out)
        self._logger.debug(f"shape after block5 {out.shape}")
        out = self.conv1(out)
        self._logger.debug(f"output shape {out.shape}")
        batch, c, d, h, w = out.shape
        out = out.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        self._logger.debug(f"output shape after permute and reshape {out.shape}")
        return out


class Colorization(nn.Module):
    def __init__(self, backbone_networks, head_network):
        super(Colorization, self).__init__()
        self._logger = self._logger = get_logger("Colorization")
        self.backbone_networks = backbone_networks
        self.head_network = head_network()
        self.sofmax = nn.Softmax(dim=1)

    def forward(self, images, num_ref=3):
        x = images
        x = self.backbone_networks(x)
        x = self.head_network(x)
        self._logger.debug(f"feature shape {x.shape}")
        sim = self.get_similarity_matrix(x, num_ref)
        return sim

    def get_similarity_matrix(self, embeddings, num_ref):
        # 3 ref images and 1 target image
        # embeddings shape (batch,64,32,32)

        batch, channels, h, w = embeddings.shape

        num_samples = num_ref + 1
        # embeddings = embeddings.reshape(-1, num_samples, channels, h * w)
        stacking = []
        end = (num_samples // 2 + 1) * -1
        start = 0

        for i in range(num_samples):
            start += 1
            end += 1
            if end == 0:
                stacking.append(embeddings[start:, :, :, :])
            else:
                stacking.append(embeddings[start:end, :, :, :])

                # embeddings = torch.stack([embeddings[:-2,:,:,:],embeddings[1:-1,:,:,:],embeddings[2:,:,:,:]])
        embeddings = torch.stack(stacking,dim=1)
        embeddings = embeddings.reshape(-1,num_samples,channels,h*w )
        self._logger.debug(f"Embedding shape {embeddings.shape}")

        refs = embeddings[:, :num_ref, :, :]
        targets = embeddings[:, num_ref:, :, :]

        # similarity (batch, num_ref, channel, h,w)
        # transpose (ref (batch ,channels,h*w*num_ref)) x target(batch,channels,h*w) (batch,64,3072) * (batch*64*1024)
        # as per paper (batch, 32x32, 64) x (batch, 64, 32x32x3)
        targets = targets.reshape(-1, channels, h * w)
        # for channel first need to transpose ref and channel first then do reshape (batch,channels,reference_pixels)
        self._logger.debug(f"ref shape before transpose {refs.shape}")
        refs = refs.permute([0, 2, 1, 3])
        self._logger.debug(f"ref shape after transpose {refs.shape}")
        refs = torch.transpose(refs.reshape(-1, channels, h * w * num_ref), 2, 1)

        self._logger.debug(f"Similarity mat multiplication refs x target   {refs.shape} x {targets.shape}")
        similarity_matrix = torch.matmul(refs, targets)
        similarity_matrix = self.sofmax(similarity_matrix)

        self._logger.debug(f"similarity matrix {similarity_matrix.shape}")
        return similarity_matrix

    def proxy_task(self, similarity_matrix, labels, num_ref=3):
        num_samples = num_ref + 1
        batch, num_classes, h, w = labels.shape

        stacking = []
        end = (num_samples // 2 + 1) * -1
        start = 0
        for i in range(num_samples):
            start += 1
            end += 1
            if end == 0:
                stacking.append(labels[start:, :, :, :])
            else:
                stacking.append(labels[start:end, :, :, :])

        labels = torch.stack(stacking,dim=1)
        labels = labels.reshape(-1, num_samples, num_classes, h * w)

        ref_labels = labels[:, :num_ref, :, :].float()
        target_labels = labels[:, num_ref:, :, :]

        # as per paper (batch,16,32x32x3)
        # for channel first need to transpose ref and channel first then do reshape (batch,color_class,reference_pixels)
        ref_labels = ref_labels.permute([0, 2, 1, 3])
        ref_labels = ref_labels.reshape(-1, num_classes, h * w * num_ref)
        target_labels = target_labels.reshape(-1, num_classes, h, w)

        # similarity_matrix = torch.transpose(similarity_matrix,1,2)
        self._logger.debug(f"reference labels shape {ref_labels.shape}, target labels shape {target_labels.shape}")

        self._logger.debug(f"color prediction  ref x similarity  {ref_labels.shape} x {similarity_matrix.shape}")
        predicted_labels = torch.matmul(ref_labels, similarity_matrix)
        predicted_labels = predicted_labels.reshape(-1, num_classes, h, w)
        self._logger.debug(f"prediction shape {predicted_labels.shape}, target labels shape {target_labels.shape} ")

        return predicted_labels, target_labels


def get_head_network(version):
    versions = {"v1": Network3D,
                "v2": Network3Dv2}
    network = versions.get(version, None)
    if not network:
        raise KeyError(f"{version} not found, available versions {list(versions.keys())}")
    return network


def get_colorization_network(backbone_name="resnet18", head_version="v1"):
    backbone_network = backbone_factory.get_backbone(backbone_name)
    head_network = get_head_network(head_version)
    colornet = Colorization(backbone_network, head_network)
    return colornet
