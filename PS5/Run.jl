# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: October, 2022

##########################################################
###################### PROGRAM RUNNER ####################
##########################################################

# Include model
include("Model.jl")

# Initialize and run
results = Initialize()
SolveModel(results, false)