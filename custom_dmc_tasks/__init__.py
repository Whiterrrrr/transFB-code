"""Custom DeepMind Control Suite tasks."""

from custom_dmc_tasks import walker
from custom_dmc_tasks import quadruped
from custom_dmc_tasks import jaco
from custom_dmc_tasks import point_mass_maze
from custom_dmc_tasks import cheetah
from custom_dmc_tasks import hopper


def make(
    domain, task, task_kwargs=None, environment_kwargs=None, visualize_reward=False
):

    if domain == "walker":
        return walker.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "point_mass_maze":
        return point_mass_maze.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "quadruped":
        return quadruped.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "jaco":
        return jaco.make(
            task,
            obs_type="perfect_features",
            seed=42,
        )
    elif domain == "cheetah":
        return cheetah.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "cartpole":
        return hopper.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    else:
        raise f"{task} not found"


def make_jaco(task, obs_type, seed):
    return jaco.make(task, obs_type, seed)
