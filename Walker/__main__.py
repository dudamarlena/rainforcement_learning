""" Main module to run learning choice robot """

from Walker import (
    walkler_ddpg,
    robot_dql,
    config,
)

if __name__ == "__main__":
    choice_robot = None
    robot_name = input('You can choose tree robots: Walker, HalfCheetah and Humanoid')
    if robot_name == 'Walker':
        choice_robot = config.WALKER
    elif robot_name == 'HalfCheetah':
        choice_robot = config.HALFCHEETAH
    elif robot_name == 'Humanoid':
        choice_robot = config.HUMANOID

    print(f'You choice robot: {robot_name}')
    algorithm_name = input('You can choose two algorithms: '
                           '\nDeep Q-Learning(DQL) '
                           '\nor Deep Deterministic Policy Gradient (DDPG). '
                           '\nWrite short name of algorithm.')
    if algorithm_name == 'DQL':
        robot_dql.main(robot_name)
    elif algorithm_name == 'DDPG':
        walkler_ddpg.main(robot_name)
