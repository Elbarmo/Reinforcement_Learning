# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:48:44 2018

@author: eia
"""
from random import randint
from numpy.linalg import inv
import numpy as np
np.set_printoptions(threshold=np.nan)

######## MC CONTROL ##########

states = np.arange(25).reshape(5,5)
action_list = ["north","south","east","west"]

terminate_state_plus = {"x":3,"y":3}
terminate_state_minus = {"x":1,"y":1}

def find_0_array(value):
    if value == 0:
        return 1
    else:
        return 0

def sample_from_policy(policy,state_ind):
    return np.random.choice(4, 1, p=policy[:,state_ind])[0]

def normalize_policy(policy):
    policy = np.fabs(policy)
    norm = np.sum(policy,axis = 0)  
    return policy/norm

def calculate_new_state(state,action):
    if action == 0:
        state["y"] -= 1
        if state["y"] == -1:
            state["y"] = 0
    if action == 1:
        state["y"] += 1
        if state["y"] == 5:
            state["y"] = 4
    
    if action == 2:
        state["x"] += 1
        if state["x"] == 5:
            state["x"] = 4
    if action == 3:
        state["x"] -= 1
        if state["x"] == -1:
            state["x"] = 0
    return state

def MC_walk(policy):
    sample = True
    while sample:
        init_state={"x":randint(0,4),"y":randint(0,4)}
        if init_state["x"] == terminate_state_plus["x"] and init_state["y"] == terminate_state_plus["y"]:
            sample = True
        elif init_state["x"] == terminate_state_minus["x"] and init_state["y"] == terminate_state_minus["y"]:
            sample = True
        else:
            sample = False
        
    count_states_action = np.zeros((4,25))
    state = init_state
    state_ind = states[state["y"],state["x"]]
    walking = True

    while walking:

        rand = randint(0,1)
        if rand == 0:
            action = randint(0,3)
        else:
            action = policy[:,state_ind].tolist().index(max(policy[:,state_ind]))
        count_states_action[action,state_ind] += 1
        
        state = calculate_new_state(state,action)
        state_ind = states[state["y"],state["x"]]
        if init_state["x"] == terminate_state_plus["x"] and init_state["y"] == terminate_state_plus["y"]:
            if debug:
                print("good")
                print(count_states_action)
            reward = np.ones((4,25))
            walking = False
        elif init_state["x"] == terminate_state_minus["x"] and init_state["y"] == terminate_state_minus["y"]:
            reward = np.zeros((4,25))
            walking = False
    
    count_states_action_0 = find_0_array_vec(count_states_action)
    policy_0_backup = np.multiply(count_states_action_0,policy)
    policy += np.divide((reward-policy),count_states_action)
    policy[policy == np.inf] = 0
    policy[policy == -np.inf] = 0
    policy = np.nan_to_num(policy)
    
    policy += policy_0_backup
    policy = normalize_policy(policy)
    policy = np.nan_to_num(policy)
    
    return policy

def e_greedy_action(state_ind,policy):
    rand = randint(0,1)
    if rand == 0:
        action = randint(0,3)
    else:
        action = policy[:,state_ind].tolist().index(max(policy[:,state_ind]))
    return action

policy = np.ones((4,25))/4
        
find_0_array_vec = np.vectorize(find_0_array)
debug = False
for i in range(100000):
    if debug:
        print("")
        print(i)
        print(policy)
    policy = MC_walk(policy)


######## TD 0 CONTROL(SARSA) ##########

def SARSA_eval(policy):
    sample = True
    while sample:
        init_state={"x":randint(0,4),"y":randint(0,4)}
        if init_state["x"] == terminate_state_plus["x"] and init_state["y"] == terminate_state_plus["y"]:
            sample = True
        elif init_state["x"] == terminate_state_minus["x"] and init_state["y"] == terminate_state_minus["y"]:
            sample = True
        else:
            sample = False
            
    count_states_action = np.zeros((4,25))
    state = init_state
    state_ind = states[state["y"],state["x"]]
    walking = True
    action = e_greedy_action(state_ind,policy)

    while walking:
            
        count_states_action[action,state_ind] += 1
        
        new_state = calculate_new_state(state,action)
        new_state_ind = states[new_state["y"],new_state["x"]]
        
        new_action = e_greedy_action(new_state_ind,policy)
        
        if new_state["x"] == terminate_state_plus["x"] and new_state["y"] == terminate_state_plus["y"]:
            
            if debug:
                print("good")
                print(count_states_action)
            reward = 1
            walking = False
            
        elif new_state["x"] == terminate_state_minus["x"] and new_state["y"] == terminate_state_minus["y"]:  
            reward = 0
            walking = False
            
        else:            
            reward = 0
            walking = True
            
        policy[action,state_ind] += np.divide((reward + policy[new_action,new_state_ind] - policy[action,state_ind]),count_states_action[action,state_ind])
        policy = normalize_policy(policy)
        policy = np.nan_to_num(policy)
        state = new_state
        state_ind = states[state["y"],state["x"]]
        action = new_action
        
    return policy

policy = np.ones((4,25))/4
debug = False
for i in range(100000):
    if debug:
        print("")
        print(i)
        print(policy)
    policy = SARSA_eval(policy)
    

######## TD Lambda CONTROL(SARSA) ##########

def SARSA_lambda_eval(policy,lambda_):
    sample = True
    while sample:
        init_state={"x":randint(0,4),"y":randint(0,4)}
        if init_state["x"] == terminate_state_plus["x"] and init_state["y"] == terminate_state_plus["y"]:
            sample = True
        elif init_state["x"] == terminate_state_minus["x"] and init_state["y"] == terminate_state_minus["y"]:
            sample = True
        else:
            sample = False
            
    count_states_action = np.zeros((4,25))
    elegibility = np.zeros((4,25))
    state = init_state
    state_ind = states[state["y"],state["x"]]
    walking = True
    action = e_greedy_action(state_ind,policy)
    while walking:
            
        count_states_action[action,state_ind] += 1
        elegibility[action,state_ind] += 1
        new_state = calculate_new_state(state,action)
        new_state_ind = states[new_state["y"],new_state["x"]]
        
        new_action = e_greedy_action(new_state_ind,policy)
        
        if new_state["x"] == terminate_state_plus["x"] and new_state["y"] == terminate_state_plus["y"]:
            
            if debug:
                print("good")
                print(count_states_action)
            reward = 1
            walking = False
            
        elif new_state["x"] == terminate_state_minus["x"] and new_state["y"] == terminate_state_minus["y"]:  
            reward = 0
            walking = False
            
        else:            
            reward = 0
            walking = True
            
        delta = reward + policy[new_action,new_state_ind] - policy[action,state_ind]
        policy += delta*np.multiply(elegibility,np.nan_to_num(np.reciprocal(count_states_action)))
        elegibility = lambda_*elegibility
        policy = normalize_policy(policy)
        
        state = new_state
        state_ind = states[state["y"],state["x"]]
        action = new_action
        
    return policy

policy = np.ones((4,25))/4
lambda_ = 0.9
debug = False
for i in range(100000):
    if debug:
        print("")
        print(i)
        print(policy)
    policy = SARSA_lambda_eval(policy,lambda_)

######## Off-line CONTROL (SARSAMAX) ##########



