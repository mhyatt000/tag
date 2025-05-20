class Registry:
    envs_cfgs: dict[str, RobotCfg]

    def __init__(self):
        self.envs_cfgs = {}
