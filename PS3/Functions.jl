# Import libraries
using Parameters, Plots, Printf, JLD2, TableView, LaTeXStrings, CSV, Tables, DataFrames

# Define parameters of the model
@with_kw struct Primitives
    β::Float64 = 0.97                                                               #Discount rate
    σ::Float64 = 2.0                                                                #Risk aversion
    α::Float64 = 0.36                                                               #Participation of capital
    δ::Float64 = 0.06                                                               #Depreciation rate
    n::Float64 = 0.011                                                              #Population growth rate
    N::Int64 = 66                                                                   #Periods
    Jᴿ::Int64 = 46                                                                  #Retirement age
    η::Array{Float64,1} = map(x->parse(Float64,x), readlines("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS3-Solution/ef.txt"))             #Age-efficiency profile
    z_state::Int64 = 2                                                              #Number of states for idiosincratic productivity
    Π::Array{Float64,2} = [0.9261 0.0739; 0.0189 0.9811]                            #Transition matrix for productivity shock
    Ψ::Array{Float64,1} = [0.2037, 0.7963]                                          #Unconditional probabilities
    a_min::Float64 = 0.0                                                              #Minimum value for asset grid
    a_max::Float64 = 75.0                                                             #Maximum value for asset grid
    a_points::Int64 = 5000                                                         #Number of points in the asset grid
    a_grid_step = range(a_min,a_max; length = a_points)                             #Asset grid step
    a_grid::Array{Float64,1} = collect(a_grid_step)                                 #Asset grid
end

# Construct vectors that will contain value functions and policy functions and parameters over which we want to make comparative statistics
mutable struct Results
    e::Array{Float64, 2}                                                            #Productivity
    θ::Float64                                                                      #Tax over labor income
    γ::Float64                                                                      #Preferences over consumption
    z::Array{Float64, 1}                                                            #Idiosyncratic productivity leves
    μ::Array{Float64}                                                               #Asset distribution
    b::Float64                                                                      #Pension benefit
    K::Float64                                                                      #Aggregate capital
    L::Float64                                                                      #Aggregate labor
    w::Float64                                                                      #Wages
    r::Float64                                                                      #Interest rate
    val_fun::Array{Float64}                                                         #Value function
    pol_fun::Array{Float64}                                                         #Policy function
    lab_sup::Array{Float64}                                                         #Labor supply
end

function Initialize(θ::Float64, z::Array{Float64, 1}, γ::Float64)
    @unpack z_state, a_points, Jᴿ, η, N = Primitives()
    e = η * z'
    val_fun = zeros(N, a_points, z_state)                                           #A matrix of zeros where column 1, 2, and 3 represents the age, asset holdings, and productivity, respectively
    pol_fun = zeros(N, a_points, z_state)                                           #A matrix of zeros where column 1, 2, and 3 represents the age, asset holdings, and productivity, respectively
    lab_sup = zeros(Jᴿ-1, a_points, z_state)                                        #Once agent retires there is no labor-leisure choice
    μ = ones(N, a_points, z_state)/sum(ones(N, a_points, z_state))                  #Distribution of assets
    L = 0.0                                                                         #Initial value for aggregate labor
    K = 0.0                                                                         #Initial value for aggregate capital
    w = 1.05                                                                        #Initial wage
    r = 0.05                                                                        #Initial interest rate
    b = 0.2                                                                         #Initial pension benefit
    Results(e, θ, γ, z, μ, b, K, L, w, r, val_fun, pol_fun,lab_sup)
end

function update_prices(results, K, L)
    @unpack Jᴿ, N, α, δ = Primitives()
    results.w = (1-α)*K^(α)*L^(-α)
    results.r = α*K^(α-1)*L^(1-α)-δ
    results.b = (results.θ*results.w*L)/sum(results.μ[Jᴿ:N, :, :])
end

# Value function iteration for both agents

# Utility for retired agent
function u_ret(c::Float64, γ::Float64, σ::Float64)
    if (c > 0)
        (c^((1-σ)*γ))/(1-σ)
    else
        -Inf
    end
end

function Solve_retiree(results::Results; progress::Bool = false)
    @unpack Jᴿ, N, β, z_state, σ = Primitives()
    @unpack a_min, a_max, a_points, a_grid_step, a_grid = Primitives()

    #Last period
    results.val_fun[N,:,1] = u_ret.(a_grid, results.γ, σ)
    #Backwards induction
    for j = (N-1):-1:Jᴿ
        if (progress)
            println(j)
        end
        choice_lower = 1
        #Iteration over assets today
        for i_a = 1:a_points
            #Agent budget
            budget = (1+results.r) * a_grid[i_a] + results.b
            val_previous = -Inf
            #Iteration over assets tomorrow
            for i_a_p = choice_lower:a_points
                #Compute utility
                val = u_ret(budget-a_grid[i_a_p],results.γ,σ) + β*results.val_fun[j+1,i_a_p,1]
                #If utility decreases store values and continue
                if val<val_previous
                    results.val_fun[j, i_a, 1] = val_previous
                    results.pol_fun[j, i_a, 1] = a_grid[i_a_p-1]
                    choice_lower = i_a_p - 1
                    break
                #If we are at the top of the grid, store and continue
                elseif i_a_p == a_points
                    results.val_fun[j, i_a, 1] = val
                    results.pol_fun[j, i_a, 1] = a_grid[i_a_p]
                end
                #Update val_previous to check decreasing utility
                val_previous = val
            end
        end
    end
    # Fill in for other productivity states
    for i_z = 2:z_state
        results.pol_fun[:, :, i_z] = results.pol_fun[:, :, 1]
        results.val_fun[:, :, i_z] = results.val_fun[:, :, 1]
    end
end

#Labor-leisure decision
function lab_choice(a::Float64, a_p::Float64, e::Float64, θ::Float64, γ::Float64, w::Float64, r::Float64)
    solution = (γ*(1-θ)*e*w - (1-γ)*((1+r)*a - a_p))/((1-θ)*e*w)
    min(1,max(0,solution))
end

# Utility for workers
function u_work(c::Float64, L::Float64, γ::Float64, σ::Float64)
    if (c>0 && L>=0 && L<=1)
        (((c^γ)*((1-L)^(1-γ)))^(1-σ))/(1-σ)
    else
        -Inf
    end
end

function Solve_worker(results::Results; progress::Bool = false)
    @unpack Π, Jᴿ, β, z_state, σ = Primitives()
    @unpack a_min, a_max, a_points, a_grid_step, a_grid = Primitives()
    #Backwards induction
    for j = (Jᴿ-1):-1:1
        if (progress)
            println(j)
        end
        #Iterate over productivity today
        for i_z = 1:z_state
            choice_lower = 1
            #Iterate over assets today
            for i_a = 1:a_points
                val_previous = -Inf
                #Iterate over states tomorrow
                for i_a_p = choice_lower:a_points
                    #Solve labor-leisure decision
                    L = lab_choice(a_grid[i_a], a_grid[i_a_p], results.e[j, i_z], results.θ, results.γ, results.w, results.r)
                    # Budget
                    budget = results.w*(1-results.θ)*results.e[j,i_z]*L + (1+results.r)*a_grid[i_a]
                    val = u_work(budget-a_grid[i_a_p],L,results.γ,σ)
                    #Iterates over tomorrow productivity
                    for i_z_p = 1:z_state
                        val += β*Π[i_z,i_z_p]*results.val_fun[j+1,i_a_p,i_z_p]
                    end
                    #If utility decreases store values and continue
                    if val<val_previous
                        results.val_fun[j,i_a,i_z] = val_previous
                        results.pol_fun[j,i_a,i_z] = a_grid[i_a_p-1]
                        results.lab_sup[j,i_a,i_z] = lab_choice(a_grid[i_a], a_grid[i_a_p-1], results.e[j, i_z], results.θ, results.γ, results.w, results.r)
                        choice_lower = i_a_p -1
                        break

                    #If we are at the top of the grid store and continue
                    elseif i_a_p == a_points
                    results.val_fun[j,i_a,i_z] = val
                    results.pol_fun[j,i_a,i_z] = a_grid[i_a_p]
                    results.lab_sup[j,i_a,i_z] = lab_choice(a_grid[i_a], a_grid[i_a_p], results.e[j,i_z], results.θ, results.γ, results.w, results.r)
                    end
                    #Update val_previous to check decreasing utility
                    val_previous = val
                end
            end
        end
    end
end

# Solve household problem. First retiree then worker
function Solve_problem(results::Results; progress::Bool = false)
    Solve_retiree(results)
    Solve_worker(results)
end

# Asset distribution
function Solve_μ(results::Results; progress::Bool = false)
    @unpack a_grid, N, n, z_state, Ψ, Π, a_points = Primitives()

    # Sets distribution to zero.
    results.μ = zeros(N, a_points, z_state)

    # Fills in model-age 1 with erodgic distribution of producitivities.
    results.μ[1, 1, :] = Ψ

    for j = 1:(N-1) #Loop over ages
        if progress
            println(j)
        end
        for i_a in 1:a_points #Loop over assets
            for i_z = 1:z_state
                if results.μ[j, i_a, i_z] == 0 #Skips if no mass at j, i_a, i_z
                    continue
                end
                #Finds index of assets tomorrow
                i_a_p = argmax(a_grid .== results.pol_fun[j, i_a, i_z])
                for i_z_p = 1:z_state #Loop over productivity levels tomorrow
                    results.μ[j+1, i_a_p, i_z_p] += Π[i_z, i_z_p] * results.μ[j, i_a, i_z]
                end
            end
        end
    end

    age_weights_temp = ones(N)

    for i = 1:(N-1)
        age_weights_temp[i + 1] = age_weights_temp[i]/(1+n)
    end

    age_weight = reshape(repeat(age_weights_temp/sum(age_weights_temp), a_points * z_state), N, a_points, z_state)

    results.μ = age_weight .* results.μ
end

# Market clearing
function Calculate_lab_sup(results::Results)
    @unpack Jᴿ, a_points, z_state = Primitives()

    e_3d = reshape(repeat(results.e, a_points), Jᴿ -1, a_points, z_state)

    sum(results.μ[1:(Jᴿ - 1),:,:] .* results.lab_sup .* e_3d)
end

function Calculate_cap_sup(results::Results)
    @unpack a_grid, N, z_state, a_points, N = Primitives()

    a_grid_3d = permutedims(reshape(repeat(a_grid, N * z_state), a_points, N, z_state), (2, 1, 3))

    sum(results.μ .* a_grid_3d)
end

function Solve_model(;θ::Float64 = 0.11, z::Array{Float64, 1} = [3.0, 0.5], γ::Float64 = 0.42, λ::Float64 = 0.5)
    results = Initialize(θ, z, γ)

    K_0 = 3.3
    L_0 = 0.3

    update_prices(results, K_0, L_0)

    ϵ = 0.001  # Tolerance level
    i = 0      # Counter

    while true
        i += 1
        println("Iteration #", i)

        println("Capital Demand: ", K_0)
        println("Labor Demand: ", L_0)

        Solve_problem(results)
        Solve_μ(results)

        K_1 = Calculate_cap_sup(results)
        L_1 = Calculate_lab_sup(results)

        println("Capital Supply: ", K_1)
        println("Labor Supply: ", L_1)

        diff = abs(K_0 - K_1) + abs(L_0 - L_1)

        println("Absolute Diff: ", diff)

        println("************************************")

        if diff > ϵ
            K_0 = λ * K_1 + (1 - λ) * K_0
            L_0 = λ * L_1 + (1 - λ) * L_0
            update_prices(results, K_0, L_0)
        else
            break
        end
    end

    results.K = K_0
    results.L = L_0
    results
end

# Welfare
process_results = function(results::Results)
    # calculate total welfare
    welfare = results.val_fun .* results.μ
    welfare = sum(welfare[isfinite.(welfare)])

    # Calculate moments from wealth distribution
    @unpack a_grid, N, z_state, a_points, N = Primitives()
    a_grid_3d = permutedims(reshape(repeat(a_grid, N * z_state), a_points, N, z_state), (2, 1, 3))
    Wealth_Mean = sum(results.μ .* a_grid_3d)
    Wealth_Variance = sum(results.μ .* a_grid_3d .^ 2)
    Wealth_Second_Central_Moment = Wealth_Second_Moment - Wealth_Mean^2
    Coefficient_of_Variation = Wealth_Mean / sqrt(Wealth_Second_Central_Moment)

    # create vector of summary statistics
    [results.θ, results.γ, results.z[1], results.k, results.l,
     results.w, results.r, results.b, welfare, cv]
end

function create_table(results_vector::Array{Results})
    table = DataFrames.DataFrame(Tables.table(reduce(hcat,process_results.(results_vector))'))
    rename!(table, [:theta, :gamma, :z_H, :k, :l, :w, :r, :b, :welfare, :Coefficient_of_Variation])
end
