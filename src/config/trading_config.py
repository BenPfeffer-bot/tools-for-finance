"""
Trading configuration parameters.
"""

# Trading pairs configuration
FOREX_PAIRS = ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad"]

# Account configuration
INITIAL_BALANCE = 100000.0
MAX_POSITION_SIZE = 0.1
RISK_PER_TRADE = 0.02
MAX_DRAWDOWN = 0.15

# Trading environment configuration
WINDOW_SIZE = 50
REWARD_SCALING = 1e-3
VOLATILITY_LOOKBACK = 20

# Transaction costs
BASE_TRANSACTION_COST = 0.0001
SLIPPAGE_FACTOR = 0.0001

# Training configuration
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
GAMMA = 0.99
N_EPISODES = 2000
VALIDATION_SPLIT = 0.2

# Network configuration
HIDDEN_DIM = 256
N_LAYERS = 3
DROPOUT = 0.2

# Experience replay
BUFFER_SIZE = 100000
MIN_BUFFER_SIZE = 1000
TARGET_UPDATE_FREQ = 100

# Training stability
GRADIENT_CLIP = 1.0
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

# Early stopping
PATIENCE = 50
MIN_DELTA = 0.001

# Regularization
L2_REG = 1e-5

# Data configuration
CACHE_DIR = "data/forex"
RESAMPLE_INTERVAL = "1min"

# Model paths
MODEL_SAVE_DIR = "models/forex"
FINAL_MODEL_PATH = "models/forex/final_model.pth"

# Logging configuration
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
