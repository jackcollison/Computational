using Distributed
addprocs(4)

@everywhere using Parameters, Plots, Printf, JLD2, TableView, DelimitedFiles, SharedArrays

@everywhere include("Function_CK_par.jl") #import the functions that solve our growth model

@everywhere prim, res = Initialize(θ=0.11, Zs=[3.0, 0.5], γ=0.42) # Baseline model

@time Solve_model(prim, res)
JLD2.jldsave("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS3/CK_rep.jld2", prim = prim, res = res) # save workspace in case
#JLD2.@load "/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS3/CK_rep.jld2"

## Exercise 1: Value Function Iteration

Plots.plot(range(0,50, length =1000),res.val_func_ret[:,5], label="", xlabel ="Asset") # Value function at age 50
Plots.savefig("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS3/Value50.png")

Plots.plot(range(0,50, length =1000), res.pol_func_wor[:,20,:], label = ["Good" "Bad"], xlabel = "Asset", ylabel = "Savings", legend=:topleft) # Savings function at age 20
Plots.savefig("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS3/Saving20.png")

## Exercise 2 & 3: Compute Stationary Distribution and Do Counterfactuals

# Without SS: Set θ = 0
prim_no_ss, res_no_ss = Primitives(θ=0, Zs=[3.0, 0.5], γ=0.42)
@time Solve_model(prim_no_ss, res_no_ss)

# No risk: Set Zs = [0.5, 0.5]
prim_no_z, res_no_z = Primitives(θ=0.11, Zs=[0.5, 0.5], γ=0.42)
@time Solve_model(prim_no_z, res_no_z)

prim_no_z_ss, res_no_z_ss = Primitives(θ=0, Zs=[0.5, 0.5], γ=0.42)
@time Solve_model(prim_no_z_ss, res_no_z_ss)

# Exogenous labor: Set γ = 1
prim_ex_lab, res_ex_lab = Primitives(θ=0.11, Zs=[3.0, 0.5], γ=1)
@time Solve_model(prim_ex_lab, res_ex_lab)

prim_ex_lab_no_ss, res_ex_lab_no_ss = Primitives(θ=0, Zs=[3.0, 0.5], γ=1)
@time Solve_model(prim_ex_lab_no_ss, res_ex_lab_no_ss)
