import sys
import logging
import gym

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")
import cyberbattle._env.cyberbattle_env
gym_env = gym.make('CyberBattleChain-v0')

for i_episode in range(1):
    observation = gym_env.reset()

    total_reward = 0

    for t in range(5600):
        action = gym_env.sample_valid_action()

        observation, reward, done, info = gym_env.step(action)

        total_reward += reward

        if reward > 0:
            print('####### rewarded action: {action}')
            print(f'total_reward={total_reward} reward={reward}')
            gym_env.render()

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

    gym_env.render()

gym_env.close()
print("simulation ended")
