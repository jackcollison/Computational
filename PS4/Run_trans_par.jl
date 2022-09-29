#using Distributed
#addprocs(4)

using Parameters, Plots, Printf, JLD2, TableView, DelimitedFiles, SharedArrays

## Problem 1: Get stationary equlibria
include("Function_CK_trans_par.jl") #import the functions that solve our growth model

prim, res = Initialize(0.11, [3.0; 0.5], 0.42) # Baseline model
@time Solve_model(prim, res)

prim_no_ss, res_no_ss = Initialize(0.0, [3.0; 0.5], 0.42) # No social security model
@time Solve_model(prim_no_ss, res_no_ss)

JLD2.jldsave("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS4/CK_trans_rep.jld2", prim = prim, res = res, prim_no_ss = prim_no_ss, res_no_ss = res_no_ss) # save workspace in case

## Problem 2: Get transition dynamics
# Exit Julia and run again
include("Function_trans_non_vex.jl")
JLD2.@load "/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS4/CK_trans_rep.jld2"

V_SS_T_ret = Array(res_no_ss.val_func_ret)
V_SS_T_wor = Array(res_no_ss.val_func_wor)
Psi_SS_0_ret = Array(res.psi_ret)
Psi_SS_0_wor = Array(res.psi_wor)
K_SS_0 = res.K
K_SS_T = res_no_ss.K

prim_tran, res_tran = Initialize(0.11, [3.0; 0.5], 0.42, K_SS_0, K_SS_T, V_SS_T_wor, V_SS_T_ret, Psi_SS_0_wor, Psi_SS_0_ret)
@time Solve_trans(prim_tran, res_tran)
