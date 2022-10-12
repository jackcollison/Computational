include("model_new.jl")

# Load libraries
using CSV

# Structure for transition
mutable struct Transition
    N_t::Int64                           # length of transition
    θ::Array{Float64, 1}                 # Tax level throughout time
    k_demand_path::Array{Float64, 1}     # path of capital demand
    l_demand_path::Array{Float64, 1}     # path of labor demand
    w_path::Array{Float64, 1}            # path of wages
    r_path::Array{Float64, 1}            # path of interest rates
    b_path::Array{Float64, 1}            # path of ss benefits
    val_fun::Array{Float64}              # value functions (4 dimensions: age, assets, productivity, time)
    pol_fun::Array{Float64}              # asset policy functions (4 dimensions: age, assets, productivity, time)
    lab_supp::Array{Float64}             # optimal labor supply (4 dimensions: age, assets, productivity, time)
    μ::Array{Float64}                    # asset distribution (4 dimensions: age, assets, productivity, time)
    ss_ini::Results                      # Initial steady state
    ss_fin::Results                      # Terminal steady state
    ev::Array{Float64}                   # Consumption equivalent variation
    vote_share::Float64                  # Proportion of people in favor of change
end

function Initialize_transition(ss_ini::Results, ss_fin::Results, k_demand_path::Array{Float64}, l_demand_path::Array{Float64}, implementation_date::Int64, N_t::Int64)
    @unpack α, δ, N, a_length, z_length, Jᴿ = Primitives()

    # θ changes at the implementation_date
    θ = vcat(repeat([ss_ini.θ], implementation_date-1), repeat([ss_fin.θ], N_t - implementation_date + 1))
    val_fun       = zeros(N, a_length, z_length, N_t)
    ev            = zeros(N, a_length, z_length, N_t)
    pol_fun       = zeros(N, a_length, z_length, N_t)
    lab_supp      = zeros(Jᴿ-1, a_length, z_length, N_t)
    μ             = ones(N, a_length, z_length, N_t) / sum(ones(N, a_length, z_length))

    # Fill in value function, policy function, and labor supply at the end to be the terminal steady state.
    val_fun[:, :, :, N_t] = ss_fin.val_fun
    pol_fun[:, :, :, N_t] = ss_fin.pol_fun
    lab_supp[:, :, :, N_t] = ss_fin.lab_supp

    # Fill in distribution at the beginning to be that of the initial steady state.
    μ[:, :, :, 1] = ss_ini.μ

    # calculate prices
    w_path = (1 - α) .* k_demand_path .^ α .* l_demand_path .^ (- α)
    r_path = α .* k_demand_path .^ (α - 1) .* l_demand_path .^ (1-α) .- δ
    b_path = (θ .* w_path .* l_demand_path) ./ reshape(sum(μ[Jᴿ:N, :, :, :], dims = [1, 2, 3]), N_t)

    Transition(N_t, θ, k_demand_path, l_demand_path, w_path, r_path, b_path, val_fun, pol_fun, lab_supp, μ, ss_ini, ss_fin, ev, 0.0)
end

# Transition for households

function Solve_retiree_transition(transition::Transition; progress::Bool = false)
    @unpack β, N, Jᴿ, z_length, σ, γ = Primitives()
    @unpack a_min, a_max, a_length, a_grid, a_grid_srl = Primitives()

    # In last period of life, agents consume everything.
    transition.val_fun[N, :, 1, :] = repeat(u_ret.(a_grid, γ, σ)', transition.N_t)'

    # Backward induction starting at before being in terminal steady state
    for i_t = (transition.N_t-1):-1:1
        if (progress)
            println(i_t)
        end

        # Iterate over ages
        for j = Jᴿ:(N-1)

            choice_lower = 1 # exploits monotonicity of policy function

            # iterates over assets today
            for i_a = 1:a_length

                # budget for savings and consumption
                budget = (1 + transition.r_path[i_t]) * a_grid[i_a] + transition.b_path[i_t]

                val_previous = -Inf # stops grid search when value starts to decrease

                # iterates over assets tomorrow
                for i_a_p = choice_lower:a_length

                    # calculates utility
                    val = u_ret(budget - a_grid[i_a_p], γ, σ) + β * transition.val_fun[j+1,i_a_p,1,i_t+1]

                    # if utility starts to decrease, we save values and move on
                    if val < val_previous
                        transition.val_fun[j, i_a, 1, i_t] = val_previous
                        transition.pol_fun[j, i_a, 1, i_t] = a_grid[i_a_p-1]
                        choice_lower = i_a_p - 1
                        break

                    # if we're at the top of the grid, we save and move on
                    elseif i_a_p == a_length
                        transition.val_fun[j, i_a, 1, i_t] = val
                        transition.pol_fun[j, i_a, 1, i_t] = a_grid[i_a_p]
                    end

                    # update val_previous to check if utility is starting to decrease.
                    val_previous = val
                end
            end
        end
    end

    # Fill in for other productivity states (doesn't matter in retirement)
    for i_z = 2:z_length
        transition.pol_fun[Jᴿ:N, :, i_z, :] = transition.pol_fun[Jᴿ:N, :, 1, :]
        transition.val_fun[Jᴿ:N, :, i_z, :] = transition.val_fun[Jᴿ:N, :, 1, :]
    end
end


# Solves workers problem. Need to solve retiree problem first.
function Solve_worker_transition(transition::Transition; progress::Bool = false)
    @unpack β, Π, Jᴿ, z_length, σ, γ, e = Primitives()
    @unpack a_min, a_max, a_length, a_grid, a_grid_srl = Primitives()

    # Backward induction starting at terminal steady state.
    for i_t = (transition.N_t-1):-1:1
        if (progress)
            println(i_t)
        end

        # iterates over ages
        for j = 1:(Jᴿ-1)

            # Iterate over productivity today.
            for i_z = 1:z_length

                choice_lower = 1 # exploits monotonicity of policy function

                # Iterate over assets today.
                for i_a = 1:a_length

                    val_previous =  -Inf # stops grid search when value starts to decrease

                    # Iterate over assets tomorrow.
                    for i_a_p = choice_lower:a_length

                        # Solve labor decision
                        l = lab_choice(a_grid[i_a], a_grid[i_a_p], e[j, i_z],
                                           transition.θ[i_t], γ, transition.r_path[i_t],
                                           transition.w_path[i_t])

                        # Get budget for savings and consumption
                        budget = transition.w_path[i_t] * (1.0-transition.θ[i_t]) * e[j,i_z]*l + (1+transition.r_path[i_t])*a_grid[i_a]

                        val = u_work(budget-a_grid[i_a_p], l, γ, σ) # Instanteous utility

                        # iterates over tomorrow productivity to add continuation value
                        for i_z_p = 1:z_length
                            val += β * Π[i_z,i_z_p] * transition.val_fun[j+1,i_a_p,i_z_p, i_t + 1]
                        end

                        # if utility starts to decrease, we save values and move on
                        if val < val_previous
                            transition.val_fun[j, i_a, i_z, i_t] = val_previous
                            transition.pol_fun[j, i_a, i_z, i_t] = a_grid[i_a_p-1]
                            transition.lab_supp[j,i_a,i_z, i_t]=lab_choice(a_grid[i_a],
                                       a_grid[i_a_p-1], e[j, i_z], transition.θ[i_t], γ, transition.r_path[i_t], transition.w_path[i_t])
                            choice_lower = i_a_p - 1
                            break

                        # if we're at the top of the grid, we save and move on
                        elseif  i_a_p == a_length
                            transition.val_fun[j, i_a, i_z, i_t] = val
                            transition.pol_fun[j, i_a, i_z, i_t] = a_grid[i_a_p]
                            transition.lab_supp[j, i_a, i_z, i_t] = lab_choice(a_grid[i_a],
                                a_grid[i_a_p], e[j, i_z], transition.θ[i_t], γ, transition.r_path[i_t], transition.w_path[i_t])
                        end

                        # update val_previous to check if utility is starting to decrease.
                        val_previous = val
                    end
                end
            end
        end
    end
end

function Solve_problem_transition(transition::Transition; progress = false)

    Solve_retiree_transition(transition; progress)

    Solve_worker_transition(transition; progress)
end


# Asset distribution

function Solve_μ_transition(transition::Transition; progress::Bool = false)
    @unpack a_grid, N, n, z_length, Π₀, Π, a_length = Primitives()

    # Solve for age weights
    age_weights_temp = ones(N)
    for i = 1:(N-1)
        age_weights_temp[i + 1] = age_weights_temp[i]/(1+n)
    end
    age_weight = reshape(repeat(age_weights_temp/sum(age_weights_temp), a_length * z_length), N, a_length, z_length)

    # un-normalize first period transition distirbution using age weight.
    transition.μ[:, :, :, 1] = transition.μ[:, :, :, 1] ./ age_weight

    # sets distribution for transition period 2 and onwards to zero.
    transition.μ[:, :, :, 2:transition.N_t] .= 0.0

    # Fills in model-age 1 with erodgic distribution of producitivities.
    transition.μ[1, 1, :, :] .= Π₀

    for i_t = 1:(transition.N_t-1)
        if progress
            println(i_t)
        end
        for j = 1:(N-1) # Iterates through model-ages
            for i_a in 1:a_length # Iterates through asset levels
                for i_z = 1:z_length
                    if transition.μ[j, i_a, i_z, i_t] == 0 # skips if no mass at j, i_a, i_z
                        continue
                    end
                    # finds index of assets tomorrow
                    i_a_p = argmax(a_grid .== transition.pol_fun[j, i_a, i_z, i_t])
                    for i_z_p = 1:z_length # iterates over productivity levels tomorrow
                        transition.μ[j+1, i_a_p, i_z_p, i_t+1] += Π[i_z, i_z_p] * transition.μ[j, i_a, i_z, i_t]
                    end
                end
            end
        end
        # renormalizes the distribution using the age_weights
        transition.μ[:, :, :, i_t] = age_weight .* transition.μ[:, :, :, i_t]
    end
    transition.μ[:, :, :, transition.N_t] = age_weight .* transition.μ[:, :, :, transition.N_t]
end

# Market clearing

function Calculate_lab_supp_path(transition::Transition)
    @unpack Jᴿ, a_length, z_length, e = Primitives()

    e_3d = reshape(repeat(e, a_length), Jᴿ -1, a_length, z_length)

    path = zeros(transition.N_t)

    for i_t = 1:transition.N_t
        path[i_t] = sum(transition.μ[1:(Jᴿ - 1),:,:, i_t] .* transition.lab_supp[:,:,:,i_t] .* e_3d)
    end

    return(path)
end

function Calculate_capital_supply_path(transition::Transition)
    @unpack a_grid, N, z_length, a_length, N = Primitives()

    a_grid_3d = permutedims(reshape(repeat(a_grid, N * z_length), a_length, N, z_length), (2, 1, 3))

    path = zeros(transition.N_t)

    for i_t = 1:transition.N_t
        path[i_t] = sum(transition.μ[:, :, :, i_t] .* a_grid_3d)
    end

    return(path)
end

# Transition paths

function Solve_transition(θ_0::Float64, θ_1::Float64,
                          k_guess_0::Float64, k_guess_1::Float64,
                          l_guess_0::Float64, l_guess_1::Float64;
                          progress::Bool = false,
                          implementation_date::Int64 = 1,
                          N_t::Int64 = 30)

    # Steady states
    println("************************************")
    println("Solve for initial steady state: ")
    ss_ini = Solve_steady_state(k_guess_0, l_guess_0; θ = θ_0, progress = true)
    println("************************************")
    println("Solve for terminal steady state: ")
    ss_fin = Solve_steady_state(k_guess_1, l_guess_1; θ = θ_1, progress = true)

    # save steady state table
    table_ss = create_table([ss_ini, ss_fin])
    CSV.write("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS4-Solution/table_ss.csv", table_ss)

    ε = 0.001
    λ = 0.5
    N_t_increment = 20

    # start with linear transition path
    k_demand_path_0 = collect(ss_ini.k .+ ((0:(N_t-1)) ./ (N_t-1)) .* (ss_fin.k - ss_ini.k))
    l_demand_path_0 = repeat([ss_fin.l], N_t)

    transition = Initialize_transition(ss_ini, ss_fin, k_demand_path_0, l_demand_path_0, implementation_date, N_t)

    println("************************************")
    println("Solve for transition path: ")

    while true # loop for determining length of transition
        i = 1

        println("************************************")
        println("Transition length: ", N_t)

        while true # loop for convergence of k and l path
            println("************************************")
            println("Iteration #", i)

            println("Computing HH value and policy functions...")
            # shooting backward to solve HH problem (value and policy functions) along transition path
            Solve_problem_transition(transition)

            println("Computing asset distribution...")
            # forward induction to solve μ along transition path
            Solve_μ_transition(transition)

            # calculate k_supply_path and l_supply_path
            l_supply_path = Calculate_lab_supp_path(transition)
            k_supply_path = Calculate_capital_supply_path(transition)

            # Tests for convergence of capital and labor demand and supply.

            error = maximum([abs.(transition.k_demand_path .- k_supply_path)./k_supply_path abs.(transition.l_demand_path .- l_supply_path)./l_supply_path])

            println("Sup norm: ", error)

            if error > ε
                println("Adjusting labor and capital demand...")

                k_demand_path_1 = (1 - λ) .* transition.k_demand_path .+ λ .* k_supply_path
                l_demand_path_1 = (1 - λ) .* transition.l_demand_path .+ λ .* l_supply_path

                transition = Initialize_transition(ss_ini, ss_fin, k_demand_path_1, l_demand_path_1, implementation_date, N_t)
                i += 1
            else
                println("Capital and labor supply and demand converged.")
                break
            end
        end

        # tests for convergence at end of transition path

        error = abs(transition.k_demand_path[N_t] - ss_fin.k)/ss_fin.k

        println("************************************")
        println("Sup norm at end of transition: ", error)

        if error > ε

            println("************************************")
            println("Transition path too short. Lengthening transition path...")

            # For capital, adds straight line from the end of the previous guess to the steady state value.
            k_demand_path_extension = collect(transition.k_demand_path[N_t] .+ ((0:(N_t_increment-1)) ./ (N_t_increment-1)) .* (ss_fin.k - transition.k_demand_path[N_t]))

            # For labor, adds terminal steady state values.
            l_demand_path_extension = repeat([ss_fin.l], N_t_increment + 1)
            k_demand_path_0 = vcat(transition.k_demand_path, k_demand_path_extension)
            l_demand_path_0 = vcat(transition.l_demand_path[1:N_t - 1], l_demand_path_extension)

            N_t += N_t_increment

            transition = Initialize_transition(ss_ini, ss_fin, k_demand_path_0, l_demand_path_0, implementation_date, N_t)

        else

            println("************************************")
            println("Transition path long enough.")
            break
        end
    end

    # Compute consumption equivalent variation
    @unpack σ, γ, N = Primitives()

    transition.ev = (transition.val_fun[:, :, :, 1] ./ ss_ini.val_fun).^(1/(γ * (1 - σ)))
    transition.vote_share = sum((transition.ev .> 1) .* ss_ini.μ)

    return(transition)
end
