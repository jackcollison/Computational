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
#include("Function_trans_non_vex.jl")
include("Function_trans_non_vex_par.jl")
JLD2.@load "/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS4/CK_trans_rep.jld2"

V_SS_T_ret = Array(res_no_ss.val_func_ret)
V_SS_T_wor = Array(res_no_ss.val_func_wor)
Psi_SS_0_ret = Array(res.psi_ret)
Psi_SS_0_wor = Array(res.psi_wor)
K_SS_0 = res.K
K_SS_T = res_no_ss.K

prim_tran, res_tran = Initialize(0.11, [3.0; 0.5], 0.42, K_SS_0, K_SS_T, V_SS_T_wor, V_SS_T_ret, Psi_SS_0_wor, Psi_SS_0_ret)
@time Solve_trans(prim_tran, res_tran)
JLD2.jldsave("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS4/CK_trans_path_rep.jld2", prim_tran = prim_tran, res_tran = res_tran)

res_tran.Ks

EV = zeros(prim_tran.N)

for j in 1:prim_tran.N
    for i in 1:prim_tran.length_a_grid
        if j < prim_tran.Jr
            for k = 1:2
            EV[j] += (res_tran.val_func_wor_tran[i,j,k] / res.val_func_wor[i,j,k])^(1/(res_tran.γ * (1 - prim_tran.σ))) * Psi_SS_0_wor[i,j,k] * prim_tran.mu[j]
            end
        elseif j >= prim_tran.Jr
            EV[j] += (res_tran.val_func_ret_tran[i,j-prim_tran.Jr+1] / res.val_func_ret[i,j-prim_tran.Jr+1])^(1/(res_tran.γ * (1 - prim_tran.σ))) * Psi_SS_0_ret[i,j-prim_tran.Jr+1] * prim_tran.mu[j]
        end
    end
end
Plots.plot(1:prim_tran.N, EV, xlabel = "Age", legend = false, ylabel = "EV")
Plots.plot!(1:prim_tran.N, zeros(prim_tran.N), linetypes =:dot)
