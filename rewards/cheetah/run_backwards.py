
from custom_dmc_tasks.cheetah import run_backward

reward_function = run_backward()._task.get_reward  # pylint: disable=protected-access
