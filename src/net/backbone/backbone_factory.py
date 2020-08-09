from .resnet import get_resnet

backbones_factory = {"resnet18": get_resnet("18")}


def get_backbone(name):
    try:
        backbone = backbones_factory[name]
    except KeyError:
        raise KeyError(f"backbone {name} not found")
    return backbone
