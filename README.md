# Reinforcement Learning
## Motivation
My exercises while learning Reiforcement Learning. I use as a guide the lectures from [David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&index=1&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ). While viewing the lectures I came up with execises to practice the teachings in the lecture, so I decided to implement them. 
## Contents
This repository contains the following examples:
1. Labyrinth_Problem.py: requires labyrinth1.json. The goal of this script is that taking a labyrinth schema, the script will estimate the value of each posible state (position in the labyrinth). Once it has converged, given any state, the way out of it will be moving into states with smaller value. (Need code cleaning) 
1. Random_Walk_Predictions.py: This script will evaluate the value of the steps in a random walk (only one degree of freedom) using a MonteCarlo method, the TD(0) and the TD(lambda).  (Need code cleaning)
1. Wind_Grid_Control.py: This script will extract the optimal policy for a Grid World walk. The policy will be optained using MonteCarlo, SARSA and SARSAmax (Need code cleaning and adding the Wind to the Grid configuration)
