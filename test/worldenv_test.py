from tag.gym.envs.world import WorldEnv, WorldEnvConfig
import tag.gym.base.config as config

import genesis as gs
import numpy as np

############# Initialization #############
gs.init(backend=gs.gpu)
cfg = WorldEnvConfig()
env = WorldEnv(cfg)

############# WorldEnv initialization #############
env.build()

############# Run #############
for i in range(400):
    env.step()
    env.set_camera(pos = (16.0 * np.sin(i / 100), 16.0 * np.cos(i / 100), 6.5))
    env.render()
env.record_visualization("worldenv_test")