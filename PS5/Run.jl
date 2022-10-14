## Run Krusell-Smith Model

include("KS_Functions.jl")

pars, grids, shocks, res = Initialize()
@time run_KS(pars, grids, shocks, res)
