# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 09:14:59 2018

@author: eia
"""
from random import randint, uniform
import matplotlib.pyplot as plt
import numpy as np


size_walk = 50
reward_one = 23
reward_two = 43
actions = 2

def turn_state_to_array(state):
    row = state[0]
    col = state[1]
    array = np.zeros((actions*size_walk,1))

    array[2*row+col,0] = 1
    return array

def calc_(state_action,policy):
    x_array = turn_state_to_array(state_action)
    
    next_state = get_next_state(state_action[0],state_action[1])
    
    next_action = sampled_policy(policy[next_state[0]][1])
    
    next_x_array = turn_state_to_array((next_state[0],next_action))
    
    return x_array*(x_array-0.9*next_x_array).T, x_array*next_state[1]
    
def world_response(state,action):
    world_naughtyness = randint(0,9)
    if world_naughtyness == 0:
        action =-1*action
    return action
    
def rand_policy():
    return randint(0,size_walk-1)

def sampled_policy(state_policy):
    sampled = uniform(0,1)
    if sampled<state_policy:
        return 0
    else:
        return 1

def max_policy(state_policy):
    if state_policy < 0.5:
        return 0
    else:
        return 1 
    
def get_next_state(state,action):
    if action == 0:
        action = -1
    action = world_response(state,action)    
    state += action
    if state == -1:
        state = 0
    elif state == 50:
        state = 49
        
    if state == reward_one or state == reward_two:
        reward = 1
    else:
        reward = 0
    return state,reward

def random_walk(numIter):
    D = []
    state = rand_policy()
    for i in range(numIter):
        action = randint(0,1)
        chosen_action = action
        state,reward = get_next_state(state,chosen_action)
        D.append((state,chosen_action,reward))
    return D
        
    
if __name__ == "__main__":
    
    #### Least-Square-Policy-Iteration Temporary-Diference
    
    num_walk_Iter = 10000
    experience = random_walk(num_walk_Iter)
    
    policy = []
    for i in range(size_walk):
        policy.append((i,0.5))
        
    num_least_square_Iter = 10
    for i in range(num_least_square_Iter):
        numerand_ = 0
        numerator_ = 0
        for data in experience:
            numerand, numerator= calc_(data,policy)
            numerand_ += numerand
            numerator_ += numerator
        
        w = np.dot(np.linalg.inv(numerand_),numerator_)
        plt.plot(w[::2], color = "red", label = "left")
        plt.plot(w[1::2], color = "blue",label = "rigth")
        plt.legend()
        plt.show()
        
        for state in policy:
            right = w[2*state[0]]
            left = w[2*state[0]+1]
            action = w[2*state[0]]/(w[2*state[0]]+w[2*state[0]+1])
            policy[state[0]] = (state[0],action)
        print(policy)
    
    
    
    
    