using DataFrames, StatFiles, LinearAlgebra, Optim
include("ps1_functions.jl")

# read data
df = DataFrame(load("JF/PS1/Mortgage_performance_data.dta"))
names(df)[1]
for i = 1:ncol(df)
    println(i, " : ", names(df)[i])
end

# extract variables in X
df_X = df[:, ["i_large_loan", "i_medium_loan", "rate_spread", "i_refinance", "age_r", "cltv", "dti", "cu", "first_mort_r", "score_0", "score_1", "i_FHA", "i_open_year2", "i_open_year3", "i_open_year4", "i_open_year5"]]
X = Matrix(df_X)
X = Float64.(X)
N = size(X)[1] # number of observations
X = hcat(ones(N), X) # add column of constant

# extract variable of Y
Y = df[:, "i_close_first_year"]
Y = Float64.(Y)

# compute log-likelihood, score, and Hessian
K = size(X)[2] # number of parameters
β_init = vcat(-1.0, zeros(K-1))
LL_init = LL(β_init, Y, X)
Score_init = Score(β_init, Y, X)
Hessian_init = Hessian(β_init, Y, X)
println("LL: ", round(LL_init, digits=3))
println("Score:\n", round.(Score_init, digits=3))
println("Hessian:\n", round.(Hessian_init, digits=3))

# compute score and Hessian numerically
Score_init_num = Score_numerical(β_init, Y, X)
Hessian_init_num = Hessian_numerical(β_init, Y, X)
println("Score:\n", round.(Score_init_num, digits=3))
println("Hessian:\n", round.(Hessian_init_num, digits=3))

# compare analytical and numerical score/Hessian
Diff_score = sqrt(sum((Score_init - Score_init_num).^2)) / K
Diff_Hessian = sqrt(sum((Hessian_init - Hessian_init_num).^2)) / K^2

# Newton algorithm
@elapsed β_ML = Newton(β_init, Y, X)
println("β_ML: ", round.(β_ML, digits=3))

# optimize using Optim
@elapsed opt_bfgs = Optim.optimize(β -> -LL(β, Y, X), β_init, method=LBFGS())
@elapsed opt_simplex = Optim.optimize(β -> -LL(β, Y, X), β_init, method=NelderMead())
opt_bfgs.minimum
opt_simplex.minimum

β_bfgs = opt_bfgs.minimizer
β_simplex = opt_simplex.minimizer

# compare solutions
Diff_β_bfgs = sqrt(sum((β_ML - β_bfgs).^2)) / K
Diff_β_simplex = sqrt(sum((β_ML - β_simplex).^2)) / K
println("Difference with optimal value by BFGS:", round(Diff_β_bfgs, digits=4))
println("Difference with optimal value by Simplex:", round(Diff_β_simplex, digits=4))