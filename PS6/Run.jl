## Run Baseline deterministic model

using JLD2

include("Hopenhayn_Baseline.jl")

pars_10, res_10 = Initialize(10.0)
@time SolveModel(pars_10, res_10)

pars_15, res_15 = Initialize(15.0)
@time SolveModel(pars_15, res_15)

JLD2.jldsave("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS6/HR_base.jld2", pars_10 = pars_10, res_10 = res_10, pars_15 = pars_15, res_15 = res_15) # save workspace in case

## Run Models with shock
## Exit Julia and run again
using JLD2

include("Hopenhayn_Shock.jl")

pars_shock_10_1, res_shock_10_1 = Initialize(10.0, 1.0)
@time SolveModel(pars_shock_10_1, res_shock_10_1)

pars_shock_15_1, res_shock_15_1 = Initialize(15.0, 1.0)
@time SolveModel(pars_shock_15_1, res_shock_15_1)

pars_shock_10_2, res_shock_10_2 = Initialize(10.0, 2.0)
@time SolveModel(pars_shock_10_2, res_shock_10_2)

pars_shock_15_2, res_shock_15_2 = Initialize(15.0, 2.0)
@time SolveModel(pars_shock_15_2, res_shock_15_2)

JLD2.jldsave("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS6/HR_shock.jld2",
                pars_shock_10_1 = pars_shock_10_1, res_shock_10_1 = res_shock_10_1, pars_shock_15_1 = pars_shock_15_1, res_shock_15_1 = res_shock_15_1,
                pars_shock_10_2 = pars_shock_10_2, res_shock_10_2 = res_shock_10_2, pars_shock_15_2 = pars_shock_15_2, res_shock_15_2 = res_shock_15_2) # save workspace in case

## Work on producing moments and figures
## Exit Julia and run again
using JLD2

JLD2.@load "/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS6/HR_base.jld2"
JLD2.@load "/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS6/HR_shock.jld2"
