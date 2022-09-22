using Distributed
addprocs(4)

@everywhere using Parameters, Plots, Printf, JLD2, TableView, DelimitedFiles, SharedArrays

@everywhere include("Function_CK.jl") #import the functions that solve our growth model

@everywhere prim, res = Initialize()

## Exercise 1: Value Function Iteration

@elapsed get_g(prim, res)
Plots.plot(range(0,50, length =1000),res.val_func_ret[:,5], label="", xlabel ="Asset") # Value function at age 50
Plots.savefig("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS3/Value50.png")

Plots.plot(range(0,50, length =1000), res.pol_func_wor[:,20,:], label = ["Good" "Bad"], xlabel = "Asset", ylabel = "Savings", legend=:topleft) # Savings function at age 20
Plots.savefig("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS3/Saving20.png")

## Exercise 2 & 3: Compute Stationary Distribution and Do Counterfactuals

# Initialize again before running
@elapsed Solve_model(prim, res)
