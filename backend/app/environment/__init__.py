"""Environment subpackage for 2048 and future games."""

from .game_2048 import Game2048
from .gym_2048_env import Gym2048Env

__all__ = ["Game2048", "Gym2048Env"] 