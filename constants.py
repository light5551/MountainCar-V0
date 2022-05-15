from enum import Enum
from contracts import deprecated

# OTHER
MODEL_PATH = 'model.pth'
TARGET_MODEL_PATH = 'target_model.pth'
TENSORBOARD_LOG_PATH = "logs"

@deprecated
class ModeJoystick(Enum):
    ON = True
    OFF = False
