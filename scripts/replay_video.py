"""Replay trajectory and save video, with custom env imports."""
import sys
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

from mani_skill.trajectory.replay_trajectory import main, parse_args

if __name__ == "__main__":
    main(parse_args())
