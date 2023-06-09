from datetime import datetime
import torch

# General constants
NORMAL = "Normal"
UNIFORM = "Uniform"
PYTORCH_INIT = "PyTorch"
now = datetime.now()
TIME_STAMP = now.strftime("%Y_%m_%d_%H")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# General hyper parameters
MAX_EPOCH = 200
TRAIN_EPOCH = 100
BATCH_SIZE = 64
RESERVED_SAMPLE = 0
INIT_MODE = PYTORCH_INIT
BATCH_TRAINING = True

# Data set related
PURCHASE100 = "Purchase100"
CIFAR_10 = "Pre-trained CIFAR-10"
LOCATION30 = "Location30"
TEXAS100 = "Texas100"
PURCHASE100_PATH = "./datasets-master/dataset_purchase"
CIFAR_10_PATH = "./datasets-master/cifar10_pretrained/cifar100_resnet20_"
LOCATION30_PATH = "./datasets-master/bangkok"
TEXAS100_PATH = "./datasets-master/texas100.npz"
DEFAULT_SET = LOCATION30

LABEL_COL = 0
LABEL_SIZE = 100
TRAIN_TEST_RATIO = (0.5, 0.5)

# Data set Distribution
CLASS_BASED = "CLASS_BASED"
DEFAULT_DISTRIBUTION = None

# Robust AGR
TRMEAN = "Trimmed Mean"
MULTI_KRUM = "Multi-Krum"
MEDIAN = "Median"
FANG = "Fang"
NONE = "None"
DEFAULT_AGR = MEDIAN

# Federated learning parameters
NUMBER_OF_PARTICIPANTS = 30
PARAMETER_EXCHANGE_RATE = 1
PARAMETER_SAMPLE_THRESHOLD = 1
GRADIENT_EXCHANGE_RATE = 1
GRADIENT_SAMPLE_THRESHOLD = 1

# Attacker related
NUMBER_OF_ADVERSARY = 1
NUMBER_OF_ATTACK_SAMPLES = 1024
RECORD_PER_N_ROUNDS = 200
BLACK_BOX_MEMBER_RATE = 0.5
WHITE_BOX_SHUFFLE_COPIES = 5
SCORE_BASED_STRATEGY = "Score_based"
NORM_BASED_STRATEGY = "Norm_based"
WHITE_BOX_PREDICTION_STRATEGY = SCORE_BASED_STRATEGY
WHITE_BOX_GLOBAL_TARGETED_ROUND = 100
FRACTION_OF_ASCENDING_SAMPLES = 1
NORM_SCALLING = 5
ASCENT_FACTOR = 15
ADJUST_RATE = 0.002
MISLEAD_FACTOR = 10
KEEP_CLASS = None


# IO related
EXPERIMENTAL_DATA_DIRECTORY = "./data_honest_defence/"


# Random seed
GLOBAL_SEED = 9