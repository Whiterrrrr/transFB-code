from custom_dmc_tasks.cheetah import run

reward_function = run()._task.get_reward  # pylint: disable=protected-access
