# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:15:38 2018

@author: eia
"""
import json
import random

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
    
def iter_values(states_dic):
    for state in states_dic.keys():
        value_north = value(states_dic,states_dic[state],"north")
        value_south = value(states_dic,states_dic[state],"south")
        value_east = value(states_dic,states_dic[state],"east")
        value_west = value(states_dic,states_dic[state],"west")
        
        value_mean = rand_policy_mean_value(value_north,value_south,value_east,value_west)
        states_dic[state]["value"] = states_dic[state]["reward"] + value_mean
   
def iter_five(states_dic):
    for i in range(500):
        iter_values(states_dic)
        
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
    converged = False
    
    
    iteration = 0
    for i in range(500):
        iteration +=1
        
        iter_values(states_dic)
    



