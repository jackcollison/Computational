#parameters
@with_kw struct Primitives
    n::Float64 = 0.011 #population growth
    N::Int64 = 66 #maximum age
    β::Float64 = 0.97 #discount rate
    σ::Float64 = 2 #risk aversion
    Jr::Int64 = 46 #retirement age
    θ::Float64 = 0.11 #labor income tax
    γ::Float64 = 0.42 #weight on consumption
    Zs::Array{Float64} = [3.0, 0.5] #idiosyncratic productivity
    α::Float64 = 0.36 #capital share of income
    δ::Float64 = 0.06 #depreciation rate
    a_min::Float64 = 0 #asset lower bound
    a_max::Float64 = 10 #asset upper bound
    length_a_grid::Int64 = 500 #number of asset grid points
    b::Float64 = 0.2 #social security benefit
    a_grid::Array{Float64,1} = collect(range(a_min, length = length_a_grid, stop = a_max)) #asset grid
    Π::Matrix{Float64} = [0.9261 0.0739; 0.0189 0.9811] #transition matrix
    ef::Matrix{Float64} = DelimitedFiles.readdlm("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS3/ef.txt", '\n')
end

#structure that holds model results
mutable struct Results
    val_func_ret::Matrix{Float64} #value function retired
    pol_func_ret::Matrix{Float64} #policy function retired
    val_func_wor::Array{Float64, 3} #value function workers
    pol_func_wor::Array{Float64, 3} #policy function workers - saving
    lab_func_wor::Array{Float64, 3} #labor supply function workers
    mu::Array{Float64, 3}
    r::Float64  #interest rate
    w::Float64  #wage
    L::Float64  #aggregate labor
    K::Float64  #aggregate capital
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func_ret = zeros(prim.length_a_grid, prim.N-prim.Jr+1) #initial value function guess
    pol_func_ret = zeros(prim.length_a_grid, prim.N-prim.Jr+1) #initial policy function guess
    val_func_wor = zeros(prim.length_a_grid, prim.Jr-1, 2)
    pol_func_wor = zeros(prim.length_a_grid, prim.Jr-1, 2)
    lab_func_wor = zeros(prim.length_a_grid, prim.Jr-1, 2)
    mu = cat(reshape(cumprod([1; ones(65)./(1+prim.n)])./sum(cumprod([1; ones(65)./(1+prim.n)])), prim.N, 1) * reshape(ones(prim.length_a_grid)./prim.length_a_grid, 1, prim.length_a_grid) .* 0.2037,
            reshape(cumprod([1; ones(65)./(1+prim.n)])./sum(cumprod([1; ones(65)./(1+prim.n)])), prim.N, 1) * reshape(ones(prim.length_a_grid)./prim.length_a_grid, 1, prim.length_a_grid) .* 0.7963, dims=3)
    r::Float64 = 0.05 #interest rate
    w::Float64 = 1.05 #wage
    L::Float64 = 1
    K::Float64 = 3
    res = Results(val_func_ret, pol_func_ret, val_func_wor, pol_func_wor, lab_func_wor, mu, r, w, L, K) #initialize results struct
    prim, res #return deliverables
end

# T operator
function Bellman(prim::Primitives,res::Results)
    @unpack val_func_ret, val_func_wor, r, w = res #unpack value function
    @unpack a_grid, β, n, N, Jr, b, θ, σ, γ, length_a_grid, Π, Zs, ef = prim #unpack model primitives
    v_next_ret = zeros(length_a_grid, N-Jr+1) #next guess of value function to fill
    ctmp_ret = zeros(length_a_grid) #consumption matrix to fill
    vtmp_ret = zeros(length_a_grid, length_a_grid)

    ctmp_ret = (1+r) .* a_grid .+ b
    val_func_ret[:,N-Jr+1] = ctmp_ret.^((1-σ)*γ) ./ (1 - σ)
    v_next_ret[:,N-Jr+1] = val_func_ret[:,N-Jr+1]

    for a_index = 1:length_a_grid, j = collect(N-Jr:-1:1) # retirement group iteration
        a = a_grid[a_index] #value of k
        ctmp_ret = (1+r) * a .+ b .- a_grid
        ctmp_ret = ifelse.(ctmp_ret .> 0, 1, 0).*ctmp_ret

        vtmp_ret[a_index,:] = ctmp_ret.^((1-σ)*γ) ./(1-σ) .+ β .* val_func_ret[:,j+1]
        v_next_ret[a_index,j] = maximum(vtmp_ret[a_index,:])
        res.pol_func_ret[a_index,j] = a_grid[findmax(vtmp_ret[a_index,:])[2]]
    end

    v_next_wor = zeros(length_a_grid, Jr-1, 2)
    ctmp_wor = zeros(length_a_grid, 2)
    vtmp_wor = zeros(length_a_grid, length_a_grid, 2)
    ltmp_wor = zeros(length_a_grid,length_a_grid,2)

    for i = 1:length_a_grid, j = 1:2
        ltmp_wor[i,:,j] = (γ .* (1 - θ) * ef[Jr-1] .* Zs[j] .* w .- (1-γ) .* ((1+r) .* a_grid[i] .- a_grid)) ./ ((1-θ) * w * ef[Jr-1] * Zs[j])
        ltmp_wor[i,ltmp_wor[i,:,j] .< 0,j] .= 0
        ltmp_wor[i,ltmp_wor[i,:,j] .> 1,j] .= 1

        ctmp_wor[:,j] = w .* (1 - θ) .* ef[Jr-1] .* Zs[j] .* ltmp_wor[i,:,j] .+ (1 + r) .* a_grid[i] .- a_grid
        ctmp_wor[:,j] = ifelse.(ctmp_wor[:,j] .> 0, 1, 0).*ctmp_wor[:,j]
        vtmp_wor[i,:,j] = (ctmp_wor[:,j].^γ .* (1 .-ltmp_wor[i,:,j]).^(1-γ)).^(1-σ) ./ (1-σ) .+ β .* v_next_ret[:,1]

        v_next_wor[i,Jr-1,j] = maximum(vtmp_wor[i,:,j])
        res.pol_func_wor[i,Jr-1,j] = a_grid[findmax(vtmp_wor[i,:,j])[2]]
        res.lab_func_wor[i,Jr-1,j] = ltmp_wor[i,findmax(vtmp_wor[i,:,j])[2],j]
    end

    val_func_wor[:,Jr-1,:] = v_next_wor[:,Jr-1,:]

    ctmp_wor = zeros(length_a_grid, 2)
    vtmp_wor = zeros(length_a_grid, length_a_grid, 2)
    ltmp_wor = zeros(length_a_grid,length_a_grid,2)

    for i = 1:length_a_grid, j = collect(Jr-2:-1:1), k = 1:2
        a = a_grid[i]
        ltmp_wor[i,:,k] = (γ .* (1 - θ) * ef[j] .* Zs[k] .* w .- (1-γ) .* ((1+r) .* a .- a_grid)) ./ ((1-θ) * w * ef[j] * Zs[k])
        ltmp_wor[i,ltmp_wor[i,:,k] .< 0,k] .= 0
        ltmp_wor[i,ltmp_wor[i,:,k] .> 1,k] .= 1
        ctmp_wor[:,k] = w .* (1 - θ) .* ef[Jr-1] .* Zs[k] .* ltmp_wor[i,:,k] .+ (1 + r) .* a .- a_grid
        ctmp_wor[:,k] = ifelse.(ctmp_wor[:,k] .> 0, 1, 0).*ctmp_wor[:,k]

        vtmp_wor[i,:,k] = (ctmp_wor[:,k].^γ .* (1 .-ltmp_wor[i,:,k]).^(1-γ)).^(1-σ) ./ (1-σ) .+ β .* val_func_wor[:,j+1,:] * Π[k,:]

        v_next_wor[i,j,k] = maximum(vtmp_wor[i,:,k])
        res.pol_func_wor[i,j,k] = a_grid[findmax(vtmp_wor[i,:,k])[2]]
        res.lab_func_wor[i,j,k] = ltmp_wor[i,findmax(vtmp_wor[i,:,k])[2],k]
    end
    return v_next_wor, v_next_ret #return next guess of value function
end

#Value function iteration
function get_g(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter

        println("###############################################")
        println("########## SOLVING HOUSEHOLD PROBLEM ##########")
        println("###############################################\n")

    while err>tol #begin iteration
        v_next_wor, v_next_ret = Bellman(prim, res) #spit out new vectors
        err1 = maximum(abs.(v_next_wor.-res.val_func_wor) ./ abs.(res.val_func_wor))
        err2 = maximum(abs.(v_next_ret.-res.val_func_ret) ./ abs.(res.val_func_ret))
        err = maximum([err1, err2])
        res.val_func_wor = v_next_wor
        res.val_func_ret = v_next_ret #update value function
        n+=1

            @printf "HH Iteration = %-12d Error = %.5g\n" n err
    end
        println("\n******************************************************************\n")
        println("Household problem converged in ", n, " iterations!\n")
        println("******************************************************************\n")
    #println("Value function converged in ", n, " iterations.")
end


## Should work from here!

function get_mu(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    @unpack a_grid, length_a_grid, Π, β = prim
    ED = 0.01 # excess demand

        println("###############################################")
        println("######## SOLVING DISTRIBUTION PROBLEM #########")
        println("###############################################\n")

    while abs(ED) > tol
        err = 100.0
        get_g(prim, res)
        while err>tol
            mu_new = zeros(length_a_grid, 2)

            for i = 1:length_a_grid, j = 1:2
                a_new = a_grid[i] # a'
                mu_new[i,j] = sum(res.mu.*(res.pol_func .== a_new) * Π'[:,j])
            end
            err = maximum(abs.((mu_new .- res.mu)./res.mu))
            res.mu = mu_new
        end

        ED = sum(res.pol_func .* res.mu)

        if ED > tol
            q_old = res.q
            res.q = res.q + abs(ED)/10 * (1 - q_old)/2

            println("\n******************************************************************\n")
            @printf "Excess Demand = %-8.6g New Price = %.6f\n\n" ED res.q
            println("******************************************************************\n")
        elseif ED < -tol
            q_old = res.q
            res.q = res.q - abs(ED)/10 * (q_old - β)/2

            println("\n******************************************************************\n")
            @printf "Excess Demand = %-8.6g New Price = %.6f\n\n" ED res.q
            println("******************************************************************\n")
        end
    end
            println("\n******************************************************************\n")
            @printf "Excess Demand = %.6f is within threshold!\n\n" ED
            println("******************************************************************\n")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    get_mu(prim, res)
end
