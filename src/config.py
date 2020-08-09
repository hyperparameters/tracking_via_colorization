import os
import logging


class Config:
    # EXPERIMENT CONFIG
    PROJECT_NAME = "color-net"
    EXPERIMENT_NAME = "color-net_1.7.2"
    COMMENTS = "using dataset youtube only one category"

    # MODEL CONFIG
    BACKBONE = "resnet18"
    HEAD_NETWORK_VERSION = "v1"

    # TRAINING CONFIG
    RESUME = False
    MODEL_PATH = ""
    LOSS = "cross_entropy"
    DEVICE = "cuda"
    GPU_INDEX = 0
    EPOCHS = 10000
    BASE_LR = 1e-3
    MAX_LR = 1e-4
    LEARNING_SCHEDULER = "LambdaLR"
    LR_LAMBDA = lambda epoch: Config.BASE_LR if epoch > 1000 else Config.MAX_LR
    LEARNING_MOMENTUM = 0.9
    GRADIENT_CLIP_NORM = 5.0
    OPTIMIZER = "Adam"
    INPUT_SIZE = (256, 256)
    SAVE_EPOCH = 10
    VISUALIZATION_EPOCH = 5
    VISUALIZATION_COUNT = 3
    MODEL_LOGGING = "/home/jovyan/dataset/model_logging"
    TEST_EPOCH = 50

    #PREPROCESSING
    IMAGE_MEAN = [0.5] #for each channel - input is grayscale
    IMAGE_STD = [0.5] #for each channel - input is grayscale
    
    # DATA CONFIG (can set different dataset for training/testing/color_quantization
    class TrainData:
        # DATA CONFIG
        BASE_DATA_ROOT = "/home/jovyan/dataset/tracking/"
        CATEGORIES = ["Shopping mall"]
        DATASET = "youtube"
        DATA_ROOT = os.path.join(BASE_DATA_ROOT, DATASET, "videos/")
        BATCH_SIZE = 32 # 32 x 4
        SHUFFLE = True
        BATCH_PER_VIDEO = 5
        MAX_BATCHES = 500
        VIDEO_FPS = 6
        SKIP_INITIAL = 72
        REFERENCE_FRAMES = 3  # num of reference for colorization task

    class TestData:
        # DATA CONFIG
        BASE_DATA_ROOT = "/home/jovyan/dataset/tracking/"
        DATASET = "davis"
        CATEGORIES = None
        DATA_ROOT = os.path.join(BASE_DATA_ROOT, DATASET, "test_2019/JPEGImages/480p")
        BATCH_SIZE = 32
        SHUFFLE = False
        BATCH_PER_VIDEO = 1
        MAX_BATCHES = None
        VIDEO_FPS = None
        SKIP_INITIAL = 48
        REFERENCE_FRAMES = 1  # num of reference for colorization task

    class KMeansData:
        # DATA CONFIG
        BASE_DATA_ROOT = "/home/jovyan/dataset/tracking/"
        DATASET = "davis"
        CATEGORIES = None
        DATA_ROOT = os.path.join(BASE_DATA_ROOT, DATASET, "training")
        BATCH_SIZE = 1
        SHUFFLE = False
        BATCH_PER_VIDEO = 1
        MAX_BATCHES = 5000
        VIDEO_FPS = 6
        SKIP_INITIAL = 48
        REFERENCE_FRAMES = 0  # num of reference for colorization task

    # COLORIZATION
    KMEANS_FILE = "/home/jovyan/clusters2.p"
    CLUSTERS = 16
    CHANNELS = 'lab'
    QUANTIZE_CHANNELS = (1,2)
    KMEANS_SAMPLES = 100000
    KMEANS_REFIT = False

    # LOGGING
    LOG_ROOT = "./logs"
    LOG_FILE = "run.log"
    LOGGER = logging.getLogger(f"{PROJECT_NAME}_{EXPERIMENT_NAME}")

    WANDB = False
    WANDB_LOG_ROOT = "/home/jovyan/dataset/model_logs/"
    __WANDB_KEY = os.getenv("WANDB_KEY")

    def __init__(self):
        if hasattr(Config,"_config"):
            raise Exception("Config already created")
        else:
            Config._config = self
            self.setup_logger(Config.LOGGER)
            # log experiment config
            if self.WANDB:
                if not os.path.isdir(self.WANDB_LOG_ROOT):
                    os.makedirs(self.WANDB_LOG_ROOT)
            print(f"changing model logging from {Config.MODEL_LOGGING} to wandb root {Config.WANDB_LOG_ROOT}/wandb")
            Config.MODEL_LOGGING = os.path.join(self.WANDB_LOG_ROOT,"wandb")
            config_dict = self.get_config_dict(Config._config)
            Config.LOGGER.info(f"Experiment Config \n {config_dict}")
            self.display2(config_dict)
            #set wandb key
            os.environ['WANDB_API_KEY'] = Config.__WANDB_KEY
            
    def display2(self,config):
        for k,v in config.items():
            print(f"{k}\t\t\t{v}\n")

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        confs = ""
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                conf = "{:30} {}".format(a, getattr(self, a))
                print(conf)
                confs= confs+"\n" + conf
        print("\n")
        return confs
        
    @staticmethod
    def get_config():
        if not hasattr(Config, "_config"):
            Config._config = Config()
        return Config._config

    @staticmethod
    def setup_logger(logger):
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(Config.LOG_ROOT,Config.LOG_FILE))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    @staticmethod
    def get_config_dict(instance):
        config_dict = {}
        for k in dir(instance):
            v = getattr(instance,k)
            if not k.startswith("__") and not k.startswith("_"):
                if not callable(v):
                    config_dict[k]=v
                elif callable(v):
                    config_dict[k] = Config.get_config_dict(v)
        return config_dict