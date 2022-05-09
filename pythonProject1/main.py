from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make('LunarLanderContinuous-v2')
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)

agent.load_models()
np.random.seed(0)

best_score = env.reward_range[0]
score_history = []
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        env.render()
    score_history.append(score)

    # if i % 25 == 0:
    #     agent.save_models()
    avg_score = np.mean(score_history[-100:])
    if score > best_score:
        best_score = score
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 1000 games avg %.3f' % avg_score)

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)