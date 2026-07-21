import numpy as np


def _determine_whether_sparse_reward(env_name):
    # return True if the environment is sparse-reward
    # determine if the env is sparse-reward or not
    # import ipdb; ipdb.set_trace()
    if env_name in [
        "peg_insertion",
        "egg_flip",
        "ram_insertion",
        "usb_pickup_insertion",
        "object_handover",
        "libero_spatial",
    ]:
        is_sparse_reward = True
    else:
        # print(f"Warning: Environment '{env_name}' is not in dictionary. Default to non-sparse reward.")
        is_sparse_reward = False
        # raise NotImplementedError

    return is_sparse_reward


# used to calculate the MC return for sparse-reward tasks.
# Assumes that the environment issues two reward values: reward_pos when the
# task is completed, and reward_neg at all the other steps.
ENV_REWARD_INFO = {
    "peg_insertion": {  # peg insertion default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "egg_flip": {  # egg flip default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "ram_insertion": {  # ram insertion default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "usb_pickup_insertion": {  # usb pickup insertion default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "object_handover": {  # object handover default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "libero_spatial": {  # libero spatial default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
}


def _get_negative_reward(env_name, reward_scale, reward_bias):
    """
    Given an environment with sparse rewards (aka there's only two reward values,
    the goal reward when the task is done, or the step penalty otherwise).
    Args:
        env_name: the name of the environment
        reward_scale: the reward scale
        reward_bias: the reward bias. The reward_scale and reward_bias are not applied
            here to scale the reward, but to determine the correct negative reward value.

    NOTE: this function should only be called on sparse-reward environments
    """
    if "peg" in env_name:
        reward_neg = (
            ENV_REWARD_INFO["peg_insertion"]["reward_neg"] * reward_scale + reward_bias
        )
    elif "egg" in env_name:
        reward_neg = ENV_REWARD_INFO["egg_flip"]["reward_neg"] * reward_scale + reward_bias
    elif "ram" in env_name:
        reward_neg = (
            ENV_REWARD_INFO["ram_insertion"]["reward_neg"] * reward_scale + reward_bias
        )
    elif "usb" in env_name:
        reward_neg = (
            ENV_REWARD_INFO["usb_pickup_insertion"]["reward_neg"] * reward_scale + reward_bias
        )
    elif "object" in env_name:
        reward_neg = (
            ENV_REWARD_INFO["object_handover"]["reward_neg"] * reward_scale + reward_bias
        )
    elif "libero" in env_name:
        reward_neg = (
            ENV_REWARD_INFO["libero_spatial"]["reward_neg"] * reward_scale + reward_bias
        )
    else:
        raise NotImplementedError(
            """
            If you want to try on a sparse reward env,
            please add the reward_neg value in the ENV_REWARD_INFO dict.
            """
        )

    return reward_neg


def calc_return_to_go(
    env_name,
    rewards,
    masks,
    gamma,
    reward_scale=1,
    reward_bias=0,
    infinite_horizon=False,
):
    """
    Calculate the Monte Carlo return to go given a list of reward for a single trajectory.
    Args:
        env_name: the name of the environment
        rewards: a list of rewards
        masks: a list of done masks
        gamma: the discount factor used to discount rewards
        reward_scale, reward_bias: the reward scale and bias used to determine
            the negative reward value for sparse-reward environments.
        infinite_horizon: whether the MDP has inifite horizion (and therefore infinite return to go)
    """
    if len(rewards) == 0:
        return np.array([])

    # process sparse-reward envs
    is_sparse_reward = _determine_whether_sparse_reward(env_name)
    if is_sparse_reward:
        reward_neg = _get_negative_reward(env_name, reward_scale, reward_bias)

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For example, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        # sum up the rewards backwards as the return to go
        return_to_go = [0] * len(rewards)
        prev_return = 0 if not infinite_horizon else float(rewards[-1] / (1 - gamma))
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * masks[-i - 1]
            prev_return = return_to_go[-i - 1]
    return np.array(return_to_go, dtype=np.float32)

