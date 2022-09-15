using Parameters, Plots, Printf #import the libraries we want
include("Function_incomplete.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@elapsed Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func, mu = res
@unpack a_grid = prim

###### Analysis

## Policy Function

Plots.plot(a_grid, val_func, title="Value Function")
Plots.plot(a_grid, pol_func, title="Policy Functions")

## Wealth Distribution plot

Plots.plot(a_grid, res.mu, title = "Distribution")

## Lorenz Curve

## Welfare Comparison

W_fb = 1 / (1 - prim.β) * 0.9717^(1-prim.α) / (1 - prim.α) # Welfare in a complete market economy



##############Make plots
#value function

Plots.savefig("02_Value_Functions.png")

#policy functions

Plots.savefig("02_Policy_Functions.png")

    ##changes in policy function
    #pol_func_δ = copy(pol_func).-k_grid
    #Plots.plot(k_grid, pol_func_δ, title="Policy Functions Changes")
    #Plots.savefig("02_Policy_Functions_Changes.png")

println("All done!")
################################
