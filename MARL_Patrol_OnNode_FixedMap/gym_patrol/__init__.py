from gym.envs.registration import register

register(id='MAPatrol-v0',
        entry_point='gym_patrol.envs:MAPatrolEnv')