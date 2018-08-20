# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 11:19:45 2018

@author: eia
"""

import gym
env = gym.make('MountainCar-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = 0
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    
    
    
    
    