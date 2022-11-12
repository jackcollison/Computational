# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Load required libraries
using DataFrames, StatFiles, Plots, Optim

# Use model file
include("Model.jl")

# Read data
chars = DataFrame(load("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS3/Car_demand_characteristics_spec1.dta"));
ins = DataFrame(load("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS3/Car_demand_iv_spec1.dta"));
sims = Float64.(DataFrame(load("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS3/Simulated_type_distribution.dta")).Var1);

# Sort datasets to ensure index lines up
sort!(chars, [:Year, :Model_id])
sort!(ins, [:Year, :Model_id])

# Separate variables
vars = ["price","dpm","hp2wt","size","turbo","trans","Year_1986","Year_1987","Year_1988","Year_1989","Year_1990","Year_1991","Year_1992","Year_1993","Year_1994",
		"Year_1995","Year_1996","Year_1997","Year_1998","Year_1999","Year_2000","Year_2001","Year_2002","Year_2003","Year_2004","Year_2005",
		"Year_2006","Year_2007","Year_2008","Year_2009","Year_2010","Year_2011","Year_2012","Year_2013","Year_2014","Year_2015","model_class_2",
		"model_class_3","model_class_4","model_class_5","cyl_2","cyl_4","cyl_6","cyl_8","drive_2","drive_3","Intercept"]
exogenous_vars = ["dpm","hp2wt","size","turbo","trans","Year_1986","Year_1987","Year_1988","Year_1989","Year_1990","Year_1991","Year_1992","Year_1993","Year_1994",
                  "Year_1995","Year_1996","Year_1997","Year_1998","Year_1999","Year_2000","Year_2001","Year_2002","Year_2003","Year_2004","Year_2005",
		          "Year_2006","Year_2007","Year_2008","Year_2009","Year_2010","Year_2011","Year_2012","Year_2013","Year_2014","Year_2015","model_class_2",
		          "model_class_3","model_class_4","model_class_5","cyl_2","cyl_4","cyl_6","cyl_8","drive_2","drive_3","Intercept"]
ins_vars = ["i_import","diffiv_local_0","diffiv_local_1","diffiv_local_2","diffiv_local_3","diffiv_ed_0"]

# Generate matrix
X = Matrix(chars[!, vars])
Z = hcat(Matrix(chars[!, exogenous_vars]), Matrix(ins[!, ins_vars]))
markets = chars.Year

# Hacky fix for Julia bug
BLAS.set_num_threads(1)

# Contraction mapping for single market
e₁ = contraction_mapping_market(chars, 1985.0, markets, sims, 0.6)[2]
e₂ = contraction_mapping_market(chars, 1985.0, markets, sims, 0.6, newton_tol=1e-12)[2]

# Plot error over iterations
plot(e₁[2:end], label="Newton Error", color=:red)
plot!(e₂[2:end], label="Contraction Error", color=:blue)
hline!([1.0], color=:black, linestyle=:dash, label="ε = 1.0")
xlabel!("Iteration")
ylabel!("Error")

# Initialize grid and weight
λ = 0.0:0.05:1.0
W = inv(Z' * Z)
S₁ = zeros(length(λ))

# Search over grid for first stage λ
for i = 1:length(λ)
    # Compute value
    S₁[i] = gmm(chars, markets, Z, sims, λ[i], W)
end

# Extract minimum value
Ŝ₁ = minimum(S₁)
λ̂₁ = λ[S₁ .== Ŝ₁]

# Generate plot
plot(λ, S₁, label = "First-Stage Value");
scatter!([λ̂₁], [Ŝ₁], label = "First-Stage Estimate");
xlabel!("λₚ")
ylabel!("First-Stage GMM Objective")

# Compute residuals, weighting matrix with minimizer
ξ̂ = ρ(chars, markets, sims, λ̂₁[1], W)
W = inv((Z .* ξ̂)' *(Z .* ξ̂))
S₂ = zeros(length(λ))

# Search over grid for second stage λ
for i = 1:length(λ)
    # Compute value
    S₂[i] = gmm(chars, markets, Z, sims, λ[i], W)
end

# Optimization
@time λ̂₂ = optimize(λ -> gmm(chars, markets, Z, sims, λ, W), [0.5], BFGS())

# Generate plot
plot(λ, S₂, label = "Second-Stage Value");
scatter!([λ̂₂.minimizer], [λ̂₂.minimum], label = "Second-Stage Estimate");
xlabel!("λₚ")
ylabel!("GMM Objective")
savefig("second_stage.png")
