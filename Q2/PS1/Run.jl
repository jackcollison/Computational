# Include libraries
using DataFrames, StatFiles, Optim

# Import model
include("Model.jl")

# Load data
data = DataFrame(load("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS1/MortgagePerformanceData.dta"))

# Create constant column
data[!, :constant] .= 1.0

# Select relevant variables
Y = Float64.(Array(select(data, :i_close_first_year)))
X = Float64.(Array(select(data, :constant, :i_large_loan, :i_medium_loan, :rate_spread, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :score_0, :score_1, :i_FHA, :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5)))

# Useful values
N, K = size(X)

# Initialize coefficients
β = zeros(K)
β[1] = -1.0

# Log-likelihood, score, and Hessian
LogLikelihood(β, X, Y)
Score(β, X, Y)
Hessian(β, X)

# Numerical approximations
NumericalScore(β, X, Y)
NumericalHessian(β, X, Y)

# Solve via maximum likelihood
NewtonMethod(LogLikelihood, Score, Hessian, β, X, Y, verbose=true)

# Optimize via Nelder-Mead, LBFGS, and Newton
results = optimize(β -> -LogLikelihood(β, X, Y), β, NelderMead())
results = optimize(β -> -LogLikelihood(β, X, Y), g!, β, LBFGS())
results = optimize(β -> -LogLikelihood(β, X, Y), g!, h!, β, Newton())
results = optimize(β -> -LogLikelihood(β, X, Y), Optim.minimizer(results), NelderMead())
