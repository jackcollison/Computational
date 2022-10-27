using Distributed
addprocs(3)

@everywhere using Parameters, Plots #import the libraries we want
@everywhere include("02Growth_model.jl") #import the functions that solve our growth model

@everywhere prim, res = Initialize() #initialize primitive and results structs
@elapsed Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

##############Make plots
#value function
Plots.plot(k_grid, val_func, title="Value Function", label=["High Productivity" "Low Productivity"])
#Plots.savefig("02_Value_Functions.png")

#policy functions
Plots.plot(k_grid, pol_func, title="Policy Functions", label=["High Productivity" "Low Productivity"])
#Plots.savefig("02_Policy_Functions.png")

#changes in policy function
pol_func_δ = copy(pol_func).-k_grid
Plots.plot(k_grid, pol_func_δ, title="Policy Functions Changes", label=["High Productivity" "Low Productivity"])
#Plots.savefig("02_Policy_Functions_Changes.png")

println("All done!")
################################
