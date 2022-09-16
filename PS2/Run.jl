using Parameters, Plots, Printf #import the libraries we want
using JLD2, TableView
include("Function_incomplete.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@elapsed Solve_model(prim, res) #solve the model!

JLD2.jldsave("/Users/Yeonggyu/Desktop/Huggett_rep.jld2", prim = prim, res = res) # save workspace in case

@unpack val_func, pol_func, mu = res
@unpack a_grid, length_a_grid, Es = prim

###### Analysis

## Policy Function

Plots.plot(a_grid, val_func, title="Value Function", label =["Employed" "Unemployed"], legend=:topleft)
Plots.savefig("/Users/Yeonggyu/Desktop/Value Function.png")

a_max_level = a_grid[findall(a_grid .== pol_func[:,1])]
findall(a_grid .== pol_func[:,2])

Plots.plot(a_grid, pol_func, title="Policy Function", label =["Employed" "Unemployed"], legend=:topleft)
Plots.plot!(a_grid, a_grid, linestyle=:dash) # 45-degree line
Plots.savefig("/Users/Yeonggyu/Desktop/Policy Function.png")

## Wealth Distribution plot
using DataFrames

Plots.plot(a_grid, res.mu, title = "Asset Distribution", label =["Employed" "Unemployed"], legend=:topleft)
Plots.savefig("/Users/Yeonggyu/Desktop/Asset Distribution.png")

Em_mu = [a_grid.+Es[1] res.mu[:,1]]
Unem_mu = [a_grid.+Es[2] res.mu[:,2]]
Em_dat = DataFrame(wealth = Em_mu[:,1], density1 = Em_mu[:,2])
Unem_dat = DataFrame(wealth = Unem_mu[:,1], density2 = Unem_mu[:,2])
wealth_mat = outerjoin(Em_dat, Unem_dat, on=:wealth)
replace!(wealth_mat.density1, missing => 0)
replace!(wealth_mat.density2, missing => 0)
disallowmissing!(wealth_mat)

wealth_mat = Matrix(wealth_mat)
wealth_mat = sortslices(wealth_mat,dims=1,by=x->(x[1],-x[2],-x[3]),rev=false)

Plots.plot(wealth_mat[:,1], wealth_mat[:,2:3], title = "Wealth Distribution", seriestype = :line, label =["Employed" "Unemployed"], legend=:topleft)
Plots.savefig("/Users/Yeonggyu/Desktop/Wealth Distribution.png")
## Lorenz Curve

lorenz_mat = [cumsum(wealth_mat[:,1].*(wealth_mat[:,2].+wealth_mat[:,3])./sum(wealth_mat[:,1].*(wealth_mat[:,2].+wealth_mat[:,3]))) cumsum(wealth_mat[:,2].+wealth_mat[:,3])]

Plots.plot(lorenz_mat[:,2], lorenz_mat[:,1], xlabel="Fraction of Households", ylabel="Fraction of Wealth", label="", title = "Lorenz Curve")
Plots.savefig("/Users/Yeonggyu/Desktop/Lorenz Curve.png")

gini = 

## Welfare Comparison

W_fb = 1 / (1 - prim.β) * 0.9717^(1-prim.α) / (1 - prim.α) # Welfare in a complete market economy
