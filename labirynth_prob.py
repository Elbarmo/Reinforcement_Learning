# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:15:38 2018

@author: eia
"""
import json
import random

finalstate = "state28"

#class state:
#    def __init__(self, value):
#        self.value = value
#        
def rand_policy():
    direction = random.randint(0,3)
    return direction


def forward_direct_rand_policy():
    direction = random.randint(0,6)
    if direction == 6:
        return 3
    if direction == 5:
        return 2
    if direction == 4:
        return 1
    else:
        return 0
    
def rand_policy_mean_value(reward_north,reward_south,reward_east,reward_west):
    return (reward_north+reward_south+reward_east+reward_west)/4

def policy_mean_value(state,reward_north,reward_south,reward_east,reward_west):
    return state["policy"]["north"]*reward_north+state["policy"]["south"]*reward_south+\
        state["policy"]["east"]*reward_east+state["policy"]["west"]*reward_west
        
def update_policy(states_dic):
    for state in states_dic.keys():
        if state != finalstate:
            value_north = value(states_dic, states_dic[state],"north")
            value_south = value(states_dic, states_dic[state],"south")
            value_east = value(states_dic, states_dic[state],"east")
            value_west = value(states_dic, states_dic[state],"west")
            value_state = states_dic[state]["value"]
            
            percentage_north = (value_north - value_state)/value_state
            percentage_south = (value_south - value_state)/value_state
            percentage_east = (value_east - value_state)/value_state
            percentage_west = (value_west - value_state)/value_state
            
            policy_north = states_dic[state]["policy"]["north"]*(1-percentage_north)
            policy_south = states_dic[state]["policy"]["south"]*(1-percentage_south)
            policy_east = states_dic[state]["policy"]["east"]*(1-percentage_east)
            policy_west = states_dic[state]["policy"]["west"]*(1-percentage_west)
            
            norm = policy_north+policy_south+policy_east+policy_west
            
            states_dic[state]["policy"]["north"] = policy_north/norm
            states_dic[state]["policy"]["south"] = policy_south/norm
            states_dic[state]["policy"]["east"] = policy_east/norm
            states_dic[state]["policy"]["west"] = policy_west/norm

def find_state_by_coord(coord,states_dic):
    for state in states_dic.keys():
        if states_dic[state]["coord"] == coord:
            return state
        
def shift_north_coord(coord):
    return "{}{}".format(coord[0],int(coord[1])-1)
def shift_south_coord(coord):
    return "{}{}".format(coord[0],int(coord[1])+1)
def shift_east_coord(coord):
    return "{}{}".format(int(coord[0])+1,coord[1])
def shift_west_coord(coord):
    return "{}{}".format(int(coord[0])-1,coord[1])
        
def value(states_dic,stated_dic,direction):
    if stated_dic["type"][direction] == 0:
        value_value = stated_dic["value"]
    else:
        if direction == "north":
            state_new_coord =shift_north_coord(stated_dic["coord"])
        elif direction == "south":
            state_new_coord =shift_south_coord(stated_dic["coord"])     
        elif direction == "east":
            state_new_coord =shift_east_coord(stated_dic["coord"])
        elif direction == "west":
            state_new_coord =shift_west_coord(stated_dic["coord"])
            
        state_name = find_state_by_coord(state_new_coord,states_dic)
        value_value = states_dic[state_name]["value"]
        
    return value_value
    
def iter_values(states_dic,update_policy):
    for state in states_dic.keys():
        value_north = value(states_dic,states_dic[state],"north")
        value_south = value(states_dic,states_dic[state],"south")
        value_east = value(states_dic,states_dic[state],"east")
        value_west = value(states_dic,states_dic[state],"west")
        
        if update_policy == True:
            value_mean = policy_mean_value(states_dic[state],value_north,value_south,value_east,value_west)
        else:
            value_mean = rand_policy_mean_value(value_north,value_south,value_east,value_west)
        states_dic[state]["value"] = states_dic[state]["reward"] + value_mean
   
def iter_five(states_dic):
    for i in range(500):
        iter_values(states_dic,update_policy = False)
        
def iter_with_policy_update(states_dic):
    
    for state in states_dic.keys():
        states_dic[state]["policy"] = {'east': 0.25, 'north': 0.25, 'south': 0.25, 'west': 0.25}
    
    for j in range(10):
        for i in range(200):
            iter_values(states_dic, update_policy = True)
        update_policy(states_dic)
    
        
if __name__ == "__main__":
    
    
    with open('labyrinth1.json') as f:
        labyrinth = json.load(f)
        
    default_value = 0
    
    states_dic={}
    for states in labyrinth.keys():
        #states_dic[states] = state(default_value)
        labyrinth[states]["value"] = default_value
        states_dic= labyrinth
          
    ## Calculate values for each state    
    #iter_five(states_dic)
    iter_with_policy_update(states_dic)
    



