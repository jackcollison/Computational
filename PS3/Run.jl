include("Functions.jl")
using Parameters, Plots, Printf, JLD2, TableView, LaTeXStrings, CSV, Tables, DataFrames
# Question 1
@unpack a_grid = Primitives()

results = Initialize(0.11, [3.0, 0.5], 0.42)

@elapsed Solve_problem(results)

plot(a_grid, results.val_fun[50, :, 1], label = L"Value \; Function", legend=:bottomright, ylabel = L"V_{50}(a)",xlabel = L"a", c=:blue)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS3-Solution/Val_Fun_50.png")


plot([a_grid a_grid a_grid],[results.pol_fun[20, :, :] a_grid],label = [L"High" L"Low" L"45°\; Line"], legend=:bottomright, ylabel = L"a'_{20}(a,z)", xlabel = L"a", color = ["blue" "red" "black"], line = ["solid" "solid" "dash"])
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS3-Solution/Pol_Fun_20.png")


plot([a_grid a_grid], results.pol_fun[20, :, :].-a_grid, labels = [L"High" L"Low"], ylabel = L"a'_{20}(a,z)-a", color = ["blue" "red"])
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS3-Solution/Savings_Funtion_20.png")


plot([a_grid a_grid],results.lab_sup[20, :, :],labels = [L"High" L"Low"], ylabel = L"l_{20}(a,z)", xlabel = L"a", color = ["blue" "red"])
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS3-Solution/Labor_Choice_20.png")


# Question 2
# plot([a_grid a_grid a_grid], [results.μ[20, :, :] a_grid])

# Question 3
# Benchmark
@elapsed bm_ss = Solve_model()  # converges in ~9 iterations
@elapsed bm_no_ss = Solve_model(θ = 0.0)  # converges in ~11 iterations

# No productivity risk
@elapsed riskless_ss = Solve_model(z = [0.5, 0.5]) # converges in ~12 iterations
@elapsed riskless_no_ss = Solve_model(θ = 0.0, z = [0.5, 0.5], λ = 0.1)  # converges in ~52 iterations

# Inelastic labor supply
@elapsed inelastic_l_ss = Solve_model(γ = 1.0, λ = 0.8) # converges in ~6 iterations
@elapsed inelastic_l_no_ss = Solve_model(θ = 0.0, γ = 1.0, λ = 0.8) # converges in ~7 iterations

# table 1
table_1 = create_table([bm_ss, bm_no_ss,
                        riskless_ss, riskless_no_ss,
                        inelastic_l_ss, inelastic_l_no_ss])

CSV.write("table_1.csv", table_1)
