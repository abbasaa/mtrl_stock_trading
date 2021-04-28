import numpy as np
import pandas as pd

import gym
import gym_anytrading
import os
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt

TICKER = sys.argv[1]
DATA_DIR = 'Tech'

# CHECK DIR FOR FILE IF NOT THROW ERROR/RUN PREPROCESS
data_file = os.path.join(os.curdir, DATA_DIR, f'{TICKER}.csv')
df = pd.read_csv(data_file)

window_size = 250
start_index = window_size
end_index = 754

env_maker = lambda: gym.make(
    'stocks-v0',
    df = df,
    window_size = window_size,
    frame_bound = (start_index, end_index)
)

env = DummyVecEnv([env_maker])
policy_kwargs = dict(net_arch=dict(pi=[128, 128], qf=[256, 256]))
model = DQN('MlpPolicy', env, buffer_size=512, learning_starts=5000, batch_size=128, gamma=0.1,
            target_update_interval=5000, exploration_initial_eps=.9, exploration_final_eps=.1,
            policy_kwargs=policy_kwargs)
# Train the agent
model.learn(total_timesteps=504*100)
# Save the agent
model.save("dqn_lunar")


env = env_maker()
observation = env.reset()

while True:
    observation = observation[np.newaxis, ...]

    # action = env.action_space.sample()
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)

    # env.render()
    if done:
        print("info:", info)
        break

plt.figure(figsize=(16, 6))
env.render_all()
plt.show()
