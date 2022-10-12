using Plots, LaTeXStrings

include("transition_new.jl");

# Social security (before and after)
θ_0 = 0.11
θ_1 = 0.0

# Guesses for initial steady state
k_guess_0 = 3.52
l_guess_0 = 0.35

# Guesses for terminal steady state
l_guess_1 = 0.37
k_guess_1 = 4.60

# Exercise 1

@elapsed exercise_1 = Solve_transition(θ_0, θ_1, k_guess_0, k_guess_1, l_guess_0, l_guess_1; progress = true)

plot([exercise_1.r_path repeat([exercise_1.ss_ini.r], exercise_1.N_t) repeat([exercise_1.ss_fin.r], exercise_1.N_t)], label = [L"r_{t}" L"r_{ss}^{0}" L"r_{ss}^{1}"],color = ["blue" "orange" "red"], line = [:solid :dash :dash])

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/Interest_rate_1.png")

plot([exercise_1.w_path repeat([exercise_1.ss_ini.w], exercise_1.N_t) repeat([exercise_1.ss_fin.w], exercise_1.N_t)], label = [L"w_{t}" L"w_{ss}^{0}" L"w_{ss}^{1}"],color = ["blue" "orange" "red"], line =[:solid :dash :dash])

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/Wages_1.png")

plot([exercise_1.k_demand_path repeat([exercise_1.ss_ini.k], exercise_1.N_t) repeat([exercise_1.ss_fin.k], exercise_1.N_t)],label = [L"K_{t}" L"K_{ss}^{0}" L"K_{ss}^{1}"],color = ["blue" "orange" "red"], line =[:solid :dash :dash])

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/Capital_1.png")

plot([exercise_1.l_demand_path repeat([exercise_1.ss_ini.l], exercise_1.N_t) repeat([exercise_1.ss_fin.l], exercise_1.N_t)],label = [L"L_{t}" L"L_{ss}^{0}" L"L_{ss}^{1}"],color = ["blue" "orange" "red"], line =[:solid :dash :dash])

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/Labor_1.png")

plot(reshape(sum(exercise_1.ev .* exercise_1.ss_ini.μ, dims = (2, 3)), 66), label = L"EV", color = "blue")

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/EV_1.png")

exercise_1.vote_share

# Exercise 2

@elapsed exercise_2 = Solve_transition(θ_0, θ_1, k_guess_0, k_guess_1, l_guess_0, l_guess_1;
                                       progress = true, implementation_date = 21, N_t = 50)

plot([exercise_2.b_path repeat([exercise_2.ss_ini.b], exercise_2.N_t) repeat([exercise_2.ss_fin.b], exercise_2.N_t)],
     label = [L"b_{t}" L"b=0.20" L"b=0"],color = ["blue" "orange" "red"], line =[:solid :dash :dash])

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/Benefits_2.png")

plot([exercise_2.r_path repeat([exercise_2.ss_ini.r], exercise_2.N_t) repeat([exercise_2.ss_fin.r], exercise_2.N_t)],
     label = [L"r_{t}" L"r_{ss}^{0}" L"r_{ss}^{1}"],color = ["blue" "orange" "red"], line =[:solid :dash :dash])

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/Interest_rate_2.png")

plot([exercise_2.w_path repeat([exercise_2.ss_ini.w], exercise_2.N_t) repeat([exercise_2.ss_fin.w], exercise_2.N_t)],
     label = [L"w_{t}" L"w_{ss}^{0}" L"w_{ss}^{1}"],color = ["blue" "orange" "red"], line =[:solid :dash :dash])

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/Wage_2.png")

plot([exercise_2.k_demand_path repeat([exercise_2.ss_ini.k], exercise_2.N_t) repeat([exercise_2.ss_fin.k], exercise_2.N_t)],
     label = [L"K_{t}" L"K_{ss}^{0}" L"K_{ss}^{1}"],color = ["blue" "orange" "red"], line =[:solid :dash :dash])

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/Capital_2.png")

plot([exercise_2.l_demand_path repeat([exercise_2.ss_ini.l], exercise_2.N_t) repeat([exercise_2.ss_fin.l], exercise_2.N_t)],
     label = [L"L_{t}" L"L_{ss}^{0}" L"L_{ss}^{1}"],color = ["blue" "orange" "red"], line =[:solid :dash :dash])

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/Labor_2.png")

plot(reshape(sum(exercise_2.ev .* exercise_2.ss_ini.μ, dims = (2, 3)), 66),color = "blue",label = L"EV")

Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/Figures/EV_2.png")

exercise_2.vote_share
