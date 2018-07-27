# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 19:37:46 2018

@author: eia
"""

from random import randint

list_states = ["state0","stateA","stateB","stateC","stateD","stateE","state1"]

def walk(dic_states):
    walking = True
    state = list_states[randint(0,6)]
    reward = dic_states[state]["reward"]
    while walking:
        
        dic_states[state]["N"] = dic_states[state]["N"] + 1
        
        if state == "state0":
            return reward,dic_states 
        
        elif state == "state1":
            return reward,dic_states
        
        else:
            direction = randint(0,1)
            if direction == 0:
                state = list_states[list_states.index(state)-1]
                
            if direction == 1:
                state = list_states[list_states.index(state)+1]
            
            reward = reward + dic_states[state]["reward"]
            
def incremental_every_visit_policy(dic_states,reward):
    
    for state in dic_states.keys():
        if dic_states[state]["N"]> 0:
            dic_states[state]["total_N"] = dic_states[state]["total_N"] + dic_states[state]["N"]
            dic_states[state]["value"] = dic_states[state]["value"]+(reward-dic_states[state]["value"])/dic_states[state]["total_N"]
            dic_states[state]["N"] = 0
        
    return dic_states

def every_visit_policy(dic_states,reward):
     for state in dic_states.keys():
         if dic_states[state]["N"]> 0:
             dic_states[state]["value"] = dic_states[state]["value"] + reward*dic_states[state]["N"]
             dic_states[state]["total_N"] = dic_states[state]["total_N"] + dic_states[state]["N"]
             dic_states[state]["N"] = 0
     return dic_states
            
def montecarlo_runs(dic_states,num_iter):
    for i in range(num_iter):
        reward,dic_states =  walk(dic_states)
        dic_states = every_visit_policy(dic_states,reward)     
        #dic_states = incremental_every_visit_policy(dic_states,reward)
    return dic_states

num_iter = 200000
dic_states = {"state0":{"N":0,"value":0, "reward":0,"total_N":0},\
           "stateA":{"N":0,"value":0, "reward":0,"total_N":0},\
           "stateB":{"N":0,"value":0, "reward":0,"total_N":0},\
           "stateC":{"N":0,"value":0, "reward":0,"total_N":0},\
           "stateD":{"N":0,"value":0, "reward":0,"total_N":0},\
           "stateE":{"N":0,"value":0, "reward":0,"total_N":0},\
           "state1":{"N":0,"value":0, "reward":1,"total_N":0}}
montecarlo_runs(dic_states,num_iter)

for state in dic_states.keys():
    print("state: {} , value: {}".format(state,dic_states[state]["value"]/dic_states[state]["total_N"]))
    
    
#################### TD(0) ###########################

def td0_walk(dic_states):
    state = "stateC"
    walking = True
    while walking:
        direction = randint(0,1)
        if direction == 0:
            next_state = list_states[list_states.index(state)-1]
            
        elif direction == 1:
            next_state = list_states[list_states.index(state)+1]
        
        dic_states[state]["total_N"] = dic_states[state]["total_N"] + 1
        dic_states[state]["value"] = dic_states[state]["value"] + (dic_states[state]["reward"] + dic_states[next_state]["value"] - dic_states[state]["value"] )/dic_states[state]["total_N"]
    
        state = next_state
        
        if state == "state0":
            dic_states[state]["total_N"] = dic_states[state]["total_N"] + 1
            return dic_states 
        
        elif state == "state1":
            dic_states[state]["total_N"] = dic_states[state]["total_N"] + 1
            #print(dic_states)
            return dic_states
            
def td0_run(dic_states,num_iter):
    for i in range(num_iter):
        dic_states =  td0_walk(dic_states)
    return dic_states

num_iter = 50000
dic_states = {"state0":{"N":0,"value":0, "reward":0,"total_N":0},\
           "stateA":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "stateB":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "stateC":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "stateD":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "stateE":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "state1":{"N":0,"value":1, "reward":1,"total_N":0}}
td0_run(dic_states,num_iter)


###################### TD(lambda) #######################

def tdlambda_walk(dic_state,state):
    list_rewards = []
    list_values = []
    walking = True
    while walking:
        direction = randint(0,1)
        if direction == 0:
            next_state = list_states[list_states.index(state)-1]
            
        elif direction == 1:
            next_state = list_states[list_states.index(state)+1]
        
        list_rewards.append(dic_state[state]["reward"])
        list_values.append(dic_state[next_state]["value"])
        
        state = next_state
        
        if state == "state0" or state == "state1":
            return list_rewards, list_values
            
            
def tdlambda_run(dic_state,num_iter,lambda_val, debug = False):
    for j in range(num_iter):
        state = list_states[randint(0,4)+1]
        list_rewards, list_values = tdlambda_walk(dic_state,state)
        dic_state[state]["N"] += 1
        goal = 0
        if debug:
            print("")
            print(state)
            print(list_rewards)
            print(list_values)
        for i in range(len(list_rewards)):
            goal += lambda_val**(i)*(list_rewards[i]+list_values[i])
        goal = goal*(1-lambda_val)
        if debug:
            print(goal)
            print(dic_state[state]["value"])
        dic_state[state]["value"] = dic_state[state]["value"] + (goal-dic_state[state]["value"])/dic_state[state]["N"]
        if debug:
            print(dic_state[state]["value"])
    for state in dic_state.keys():
        dic_state[state]["value"] = dic_state[state]["value"]/(1-lambda_val)
    return dic_states

num_iter = 100000
lambda_val = 0.1
dic_states = {"state0":{"N":0,"value":0, "reward":0,"total_N":0},\
           "stateA":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "stateB":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "stateC":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "stateD":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "stateE":{"N":0,"value":0.5, "reward":0,"total_N":0},\
           "state1":{"N":0,"value":1, "reward":1,"total_N":0}}
tdlambda_run(dic_states,num_iter,lambda_val)







