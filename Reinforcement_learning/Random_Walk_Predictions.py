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


