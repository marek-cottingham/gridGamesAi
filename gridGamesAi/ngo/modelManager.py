from pathlib import Path
import os
import math
from time import time
from dataclasses import dataclass

from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import NGO_MODELS_DIR
import matplotlib.pyplot as plt
import numpy as np

from .gameState import NgoGameRunner, NgoGameState
from ..agents import RandomAgent
from .temporalDifferenceModel import Ngo_TD_Agent, Ngo_TD_Agent_v1b

@dataclass
class ModelManager():
    base_patch: Path = NGO_MODELS_DIR / "unnammed_model_1"
