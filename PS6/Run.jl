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
using JLD2, Plots

JLD2.@load "/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS6/HR_base.jld2"
JLD2.@load "/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS6/HR_shock.jld2"

# Prices
res_10.p
res_shock_10_1.p
res_shock_10_2.p

# Incumbents
sum(res_10.μ)
sum(res_shock_10_1.μ)
sum(res_shock_10_2.μ)

# Entrants
res_10.M
res_shock_10_1.M
res_shock_10_2.M

# Exit

sum(res_10.μ .* (res_10.X .- 1))
sum(res_shock_10_1.μ .* res_shock_10_1.Σ[:,2])
sum(res_shock_10_2.μ .* res_shock_10_2.Σ[:,2])

# Aggregate Labor

res_10.NA
res_shock_10_1.NA
res_shock_10_2.NA

# Incumbents Labor

res_10.NIm
res_shock_10_1.NIm
res_shock_10_2.NIm

# Entrants Labor

res_10.NEn
res_shock_10_1.NEn
res_shock_10_2.NEn

# Fraction of Entrants Labor

res_10.NEn / res_10.NA
res_shock_10_1.NEn / res_shock_10_1.NA
res_shock_10_2.NEn / res_shock_10_2.NA

# Plot exit decisions

Plots.plot(pars_10.S, res_10.X .- 1, xlabel = "s", label = "No Shock")
Plots.plot!(pars_10.S, res_shock_10_1.Σ[:,2], label = "α = 1")
Plots.plot!(pars_10.S, res_shock_10_2.Σ[:,2], label = "α = 2")
Plots.savefig("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS6/Exit Decisions.png")

# Compare different levels of cf

Plots.plot(pars_10.S, res_10.X .- 1, xlabel = "s", label = "No Shock, cf = 10")
Plots.plot!(pars_10.S, res_15.X .- 1, xlabel = "s", label = "No Shock, cf = 15")
Plots.plot!(pars_10.S, res_shock_10_1.Σ[:,2], label = "α = 1, cf = 10")
Plots.plot!(pars_10.S, res_shock_15_1.Σ[:,2], label = "α = 1, cf = 15")
Plots.plot!(pars_10.S, res_shock_10_2.Σ[:,2], label = "α = 2, cf = 10")
Plots.plot!(pars_10.S, res_shock_15_2.Σ[:,2], label = "α = 2, cf = 15")
Plots.savefig("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS6/Exit Decisions Comparison.png")
