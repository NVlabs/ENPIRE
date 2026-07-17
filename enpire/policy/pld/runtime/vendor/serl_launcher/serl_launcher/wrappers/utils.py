import gymnasium as gym


class DoneWrapper(gym.Wrapper):
    """
    Enables observation histories and receding horizon control.

    Accumulates observations into obs_horizon size chunks. Starts by repeating the first obs.

    Executes act_exec_horizon actions in the environment.
    """

    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)
        self.env = env

    def step(self, action, *args):
        obs, reward, done, trunc, info = self.env.step(action, *args)
        return obs, reward, done or trunc, trunc, info

