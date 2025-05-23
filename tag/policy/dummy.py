import gymnasium

class DummyPolicy:
    def __init__(self, action_space: gymnasium.Space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()
