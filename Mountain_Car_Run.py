# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 13:22:57 2018

@author: eia
"""
import numpy as np
from math import pi
import random

def random_walk(numIter,init_state):
    experience = []
    count = 0
    state = init_state
    for i in range(numIter):
        action = (random.uniform(min_velocity,max_velocity),random.uniform(0,2*pi))
        new_state = dynamics(state,action)
        if new_state[0]>0.9 and new_state[1]>0.9:
            reward = 100
            new_state = init_state
            count += 1
            print(count)
        else:
            reward = -1
            
        experience.append((state,new_state,reward))
        state = new_state
    return experience


def dynamics(state,action):
    old_vel = np.array([state[2],state[3]])
    act_vel = np.array([action[0],action[1]])
    new_vel = (old_vel+act_vel)/2
    new_state_x = state[0] + new_vel[0]*np.cos(new_vel[1])
    new_state_y = state[1] + new_vel[0]*np.sin(new_vel[1])
    
    if new_state_x>max_x_pos:
        new_state_x = min_x_pos
        
    if new_state_y>max_y_pos:
        new_state_y = min_y_pos
        
    if new_state_x<min_x_pos:
        new_state_x = max_x_pos
        
    if new_state_y<min_y_pos:
        new_state_y = max_y_pos
        
    return np.array([new_state_x,new_state_y,new_vel[0],new_vel[1]])

def calculate_value(state,weights):
    return np.dot(state,weights)[0][0]

if __name__ == "__main__":
    #### Linear function aproximation    
    #### State definition (x-value,velocity(r,theta))
    init_state = np.array([-0.5,-0.5,0.0,0.0])
    weights = np.array([[1],[1],[1],[1]])
    
    final_position = (0.9,0.9)
    
    #### Limits
    min_velocity = 0.
    max_velocity = 0.07

    min_x_pos = -1
    max_x_pos = 1
    min_y_pos = -1
    max_y_pos = 1
    
    ##### action = (r,theta)
    
    numIter = 1000000
    
    experience = random_walk(numIter,init_state)
    
    
    
    
    
    
    