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
        
    elif action == 1:
        state["y"] += 1
    
    elif action == 2:
        state["x"] += 1

    elif action == 3:
        state["x"] -= 1

    
    if state["x"]== 2:
        state["y"] -= 1
   
    elif state["x"]== 3:
        state["y"] -= 1

            
    if state["x"] < 0:
        state["x"] = 0
    if state["y"] < 0:
        state["y"] = 0
    if state["x"] > 4:
        state["x"] = 4
    if state["y"] > 4:
        state["y"] = 4
            
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

def SARSA_eval(policy,count_states_action):
    sample = True
    while sample:
        init_state={"x":randint(0,4),"y":randint(0,4)}
        if init_state["x"] == terminate_state_plus["x"] and init_state["y"] == terminate_state_plus["y"]:
            sample = True
        elif init_state["x"] == terminate_state_minus["x"] and init_state["y"] == terminate_state_minus["y"]:
            sample = True
        else:
            sample = False
            
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
count_states_action = np.zeros((4,25))
for i in range(100000):
    if debug:
        print("")
        print(i)
        print(policy)
    policy = SARSA_eval(policy,count_states_action)
    

######## TD Lambda CONTROL(SARSA) ##########

def SARSA_lambda_eval(policy,lambda_,count_states_action,count_good ,count_bad):
    sample = True
    while sample:
        init_state={"x":randint(0,4),"y":randint(0,4)}
        if init_state["x"] == terminate_state_plus["x"] and init_state["y"] == terminate_state_plus["y"]:
            sample = True
        elif init_state["x"] == terminate_state_minus["x"] and init_state["y"] == terminate_state_minus["y"]:
            sample = True
        else:
            sample = False
            
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
                #print(count_states_action)
            reward = 10
            walking = False
            count_good += 1
            
        elif new_state["x"] == terminate_state_minus["x"] and new_state["y"] == terminate_state_minus["y"]:
            if debug:
                print("bad")
                #print(count_states_action)
            reward = 0
            walking = False
            count_bad += 1
            
        else:            
            reward = 0
            walking = True
            
        delta = reward + policy[new_action,new_state_ind] - policy[action,state_ind]
        policy += delta*np.multiply(elegibility,np.nan_to_num(np.reciprocal(count_states_action)))
        elegibility = lambda_*elegibility
        #policy = normalize_policy(policy)
        
        state = new_state
        state_ind = states[state["y"],state["x"]]
        action = new_action
        
    return policy,count_good,count_bad,count_states_action

policy = np.ones((4,25))
lambda_ = 0.9
debug = False
count_good = 0
count_bad = 0
count_states_action = np.zeros((4,25))
for i in range(100000):
    if debug:
        print("")
        print(i)
        print(policy)
    policy,count_good,count_bad,count_states_action = SARSA_lambda_eval(policy,lambda_,count_states_action,count_good, count_bad)

policy = normalize_policy(policy)


def nice_display(policy):
    charar = np.chararray((5, 5),itemsize = 5)
    for state in range(25):
        row = int(state/5)
        col = state-row*5
        charar[row,col] = action_list[policy[:,state].tolist().index(max(policy[:,state]))]
    return charar
        
optimal_policy = nice_display(policy)
######## Off-line CONTROL (SARSAMAX) ##########


def SARSAmax(policy,count_states_action,count_good ,count_bad):
    sample = True
    while sample:
        init_state={"x":randint(0,4),"y":randint(0,4)}
        if init_state["x"] == terminate_state_plus["x"] and init_state["y"] == terminate_state_plus["y"]:
            sample = True
        elif init_state["x"] == terminate_state_minus["x"] and init_state["y"] == terminate_state_minus["y"]:
            sample = True
        else:
            sample = False
            
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
            
        policy[action,state_ind] += np.divide((reward + max(policy[:,new_state_ind]) - policy[action,state_ind]),count_states_action[action,state_ind])
        policy = normalize_policy(policy)
        policy = np.nan_to_num(policy)
        state = new_state
        state_ind = states[state["y"],state["x"]]
        action = new_action
        
    return policy,count_good,count_bad,count_states_action

policy = np.ones((4,25))
debug = False
count_good = 0
count_bad = 0
count_states_action = np.zeros((4,25))
for i in range(100000):
    if debug:
        print("")
        print(i)
        print(policy)
    policy,count_good,count_bad,count_states_action = SARSAmax(policy,count_states_action,count_good, count_bad)

policy = normalize_policy(policy)







































