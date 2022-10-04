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

prim_tran, res_tran = Initialize(0.11, [3.0; 0.5], 0.42, K_SS_0, K_SS_T, V_SS_T_wor, V_SS_T_ret, Psi_SS_0_wor, Psi_SS_0_ret, 30, 30)
@time Solve_trans(prim_tran, res_tran)
JLD2.jldsave("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS4/CK_trans_path_rep.jld2", prim_tran = prim_tran, res_tran = res_tran)

EV = zeros(prim_tran.N)

for j in 1:prim_tran.N
    for i in 1:prim_tran.length_a_grid
        if j < prim_tran.Jr
            for k = 1:2
            EV[j] += (res_tran.val_func_wor_tran[i,j,k] / res.val_func_wor[i,j,k])^(1/(res_tran.γ * (1 - prim_tran.σ))) * Psi_SS_0_wor[i,j,k]
            end
        elseif j >= prim_tran.Jr
            EV[j] += (res_tran.val_func_ret_tran[i,j-prim_tran.Jr+1] / res.val_func_ret[i,j-prim_tran.Jr+1])^(1/(res_tran.γ * (1 - prim_tran.σ))) * Psi_SS_0_ret[i,j-prim_tran.Jr+1]
        end
    end
end
Plots.plot(1:prim_tran.N, EV, xlabel = "Age", legend = false, ylabel = "EV")
Plots.plot!(1:prim_tran.N, zeros(prim_tran.N), linestyle =:dot, label = "")

## 엄청 나이 많은 사람들은 social security가 없어진 것을 선호하는데, 이건 노동공급 감소 + 자본 증가로 인해 전체적으로 이자율이 오르면서 이들의 소득이 증가하는 현상이 있기 때문이다.

## Exercise 2

prim_tran_ff, res_tran_ff = Initialize(0.11, [3.0; 0.5], 0.42, K_SS_0, K_SS_T, V_SS_T_wor, V_SS_T_ret, Psi_SS_0_wor, Psi_SS_0_ret, 50, 30)
@time Solve_trans(prim_tran_ff, res_tran_ff)
JLD2.jldsave("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS4/CK_trans_path_rep_cf.jld2", prim_tran_ff = prim_tran_ff, res_tran_ff = res_tran_ff)

EV2 = zeros(prim_tran_ff.N)

for j in 1:prim_tran_ff.N
    for i in 1:prim_tran_ff.length_a_grid
        if j < prim_tran_ff.Jr
            for k = 1:2
            EV2[j] += (res_tran_ff.val_func_wor_tran[i,j,k] / res.val_func_wor[i,j,k])^(1/(res_tran_ff.γ * (1 - prim_tran_ff.σ))) * Psi_SS_0_wor[i,j,k]
            end
        elseif j >= prim_tran.Jr
            EV2[j] += (res_tran_ff.val_func_ret_tran[i,j-prim_tran_ff.Jr+1] / res.val_func_ret[i,j-prim_tran_ff.Jr+1])^(1/(res_tran_ff.γ * (1 - prim_tran_ff.σ))) * Psi_SS_0_ret[i,j-prim_tran_ff.Jr+1]
        end
    end
end
Plots.plot(1:prim_tran_ff.N, EV2, xlabel = "Age", legend = false, ylabel = "EV")
Plots.plot!(1:prim_tran_ff.N, zeros(prim_tran_ff.N), linestyle =:dot, label = "")
