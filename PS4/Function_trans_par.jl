#parameters
 @with_kw struct Primitives
    n::Float64 = 0.011 #population growth
    N::Int64 = 66 #maximum age
    β::Float64 = 0.97 #discount rate
    σ::Float64 = 2 #risk aversion
    Jr::Int64 = 46 #retirement age
    α::Float64 = 0.36 #capital share of income
    δ::Float64 = 0.06 #depreciation rate
    a_min::Float64 = 0 #asset lower bound
    a_max::Float64 = 50 #asset upper bound
    length_a_grid::Int64 = 1000 #number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a_min, length = length_a_grid, stop = a_max)) #asset grid
    Π::Matrix{Float64} = [0.9261 0.0739; 0.0189 0.9811] #transition matrix
    ef::Matrix{Float64} = DelimitedFiles.readdlm("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS4/ef.txt", '\n')
    mu::Array{Float64} = cumprod([1; ones(N-1)./(1+n)])./sum(cumprod([1; ones(N-1)./(1+n)]))
    T::Int64 = 30 #transition periods
end

 mutable struct Results
    # Transition path
    val_func_ret_tran::Array{Float64, 3} #value function retired
    pol_func_ret_tran::Array{Float64, 3} #policy function retired
    val_func_wor_tran::Array{Float64, 4} #value function workers
    pol_func_wor_tran::Array{Float64, 4} #policy function workers - saving
    lab_func_wor_tran::Array{Float64, 4} #labor supply function workers

    # Prices and aggregates
    r::Float64  #interest rate
    w::Float64  #wage
    b::Float64  #social security benefit

    # Policy experiments
    θ::Float64  #labor income tax
    γ::Float64  #weight on consumption
    Zs::Array{Float64} #idiosyncratic productivity

    # Aggregate capital transitions
    Ls::Array{Float64}  #aggregate labor
    Ks::Array{Float64}

    psi_ret_tran::Array{Float64, 3}
    psi_wor_tran::Array{Float64, 4}
end

#function for initializing model primitives and results
 function Initialize(θ::Float64, Zs::Array{Float64, 1}, γ::Float64, K1::Float64, K2::Float64, Vwor::Array{Float64, 3}, Vret::Array{Float64, 2}, Psiwor::Array{Float64, 3}, Psiret::Array{Float64, 2})
    prim = Primitives() #initialize primtiives

    val_func_ret_tran = Array{Float64}(zeros(prim.length_a_grid, prim.N-prim.Jr+1, prim.T+1))  #value function retired
    pol_func_ret_tran = Array{Float64}(zeros(prim.length_a_grid, prim.N-prim.Jr+1, prim.T))  #policy function retired
    val_func_wor_tran = Array{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2, prim.T+1)) #value function workers
    pol_func_wor_tran = Array{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2, prim.T)) #policy function workers - saving
    lab_func_wor_tran = Array{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2, prim.T)) #labor supply function workers

    val_func_ret_tran[:,:,prim.T+1] = Vret
    val_func_wor_tran[:,:,:,prim.T+1] = Vwor

    psi_ret_tran = Array{Float64}(ones(prim.length_a_grid, prim.N-prim.Jr+1, prim.T+1) ./ prim.length_a_grid)
    psi_wor_tran = Array{Float64}(ones(prim.length_a_grid, prim.Jr-1, 2, prim.T+1) ./ prim.length_a_grid) #fraction of agents in each age group by state

    psi_wor_tran[:,:,1,:] = psi_wor_tran[:,:,1,:] .* 0.2037
    psi_wor_tran[:,:,2,:] = psi_wor_tran[:,:,2,:] .* 0.7963
    psi_ret_tran[:,:,1] = Psiret
    psi_wor_tran[:,:,:,1] = Psiwor

    r::Float64 = 0.05 #interest rate
    w::Float64 = 0.99 #wage
    b::Float64 = 0.2
    Ls::Array{Float64} = ones(prim.T) .* 0.5
    Ks::Array{Float64} = collect(range(K1, K2, length=prim.T+1))
    res = Results(val_func_ret_tran, pol_func_ret_tran, val_func_wor_tran, pol_func_wor_tran, lab_func_wor_tran, r, w, b, θ, γ, Zs, Ls, Ks, psi_ret_tran, psi_wor_tran) #initialize results struct
    prim, res #return deliverables
end

# T operator
 function Bellman_ret(prim::Primitives,res::Results, t::Int64)
    @unpack r, γ, b = res #unpack value function
    @unpack a_grid, β, N, Jr, σ, length_a_grid = prim #unpack model primitives
    ctmp_ret = Array{Float64}(zeros(length_a_grid)) #consumption matrix to fill
    vtmp_ret = Array{Float64}(zeros(length_a_grid))

    ctmp_ret = (1+r) .* a_grid .+ b
    res.val_func_ret_tran[:,N-Jr+1,t] = ctmp_ret.^((1-σ)*γ) ./ (1 - σ)

    for j in N-Jr:-1:1 # retirement group iteration
      for a_index = 1:length_a_grid
            a = a_grid[a_index] #value of k
            ctmp_ret = (1+r) * a .+ b .- a_grid
            ctmp_ret = ifelse.(ctmp_ret .> 0, 1, 0).*ctmp_ret

            vtmp_ret = ctmp_ret.^((1-σ)*γ) ./(1-σ) .+ β .* res.val_func_ret_tran[:,j+1,t+1]
            V = findmax(vtmp_ret)

            res.val_func_ret_tran[a_index,j,t] = V[1]
            res.pol_func_ret_tran[a_index,j,t] = a_grid[V[2]]
        end
    end
end

 function Bellman_wor(prim::Primitives, res::Results, t::Int64)
    @unpack r, w, b, θ, γ, Zs = res #unpack value function
    @unpack a_grid, β, N, Jr, σ, length_a_grid, Π, ef, T = prim #unpack model primitives

    ctmp_wor = Array{Float64}(zeros(length_a_grid))
    vtmp_wor = Array{Float64}(zeros(length_a_grid))
    ltmp_wor = Array{Float64}(zeros(length_a_grid))
    θ = θ * (t < 1)

     for (i, j) in collect(Iterators.product(1:length_a_grid, 1:2))
        ltmp_wor = (γ .* (1 - θ) * ef[Jr-1] .* Zs[j] .* w .- (1-γ) .* ((1+r) .* a_grid[i] .- a_grid)) ./ ((1-θ) * w * ef[Jr-1] * Zs[j])
        ltmp_wor[ltmp_wor.< 0] .= 0
        ltmp_wor[ltmp_wor.> 1] .= 1

        ctmp_wor = w .* (1 - θ) .* ef[Jr-1] .* Zs[j] .* ltmp_wor .+ (1 + r) .* a_grid[i] .- a_grid
        ctmp_wor = ifelse.(ctmp_wor .> 0, 1, 0).*ctmp_wor
        vtmp_wor = (ctmp_wor.^γ .* (1 .-ltmp_wor).^(1-γ)).^(1-σ) ./ (1-σ) .+ β .* res.val_func_ret_tran[:,1,t+1]

        V = findmax(vtmp_wor)

        res.val_func_wor_tran[i,Jr-1,j,t] = V[1]
        res.pol_func_wor_tran[i,Jr-1,j,t] = a_grid[V[2]]
        res.lab_func_wor_tran[i,Jr-1,j,t] = ltmp_wor[V[2]]
    end

    for j in Jr-2:-1:1
      for (i,k) in collect(Iterators.product(1:length_a_grid, 1:2))
        a = a_grid[i]
        ltmp_wor= (γ .* (1 - θ) * ef[j] .* Zs[k] .* w .- (1-γ) .* ((1+r) .* a .- a_grid)) ./ ((1-θ) * w * ef[j] * Zs[k])
        ltmp_wor[ltmp_wor .< 0] .= 0
        ltmp_wor[ltmp_wor .> 1] .= 1

        ctmp_wor = w .* (1 - θ) .* ef[j] .* Zs[k] .* ltmp_wor .+ (1 + r) .* a .- a_grid
        ctmp_wor = ifelse.(ctmp_wor.> 0, 1, 0).*ctmp_wor

        vtmp_wor = (ctmp_wor.^γ .* (1 .- ltmp_wor).^(1-γ)).^(1-σ) ./ (1-σ) .+ β .* res.val_func_wor_tran[:,j+1,:,t+1] * Π[k,:]

        V = findmax(vtmp_wor)
        res.val_func_wor_tran[i,j,k,t] = V[1]
        res.pol_func_wor_tran[i,j,k,t] = a_grid[V[2]]
        res.lab_func_wor_tran[i,j,k,t] = ltmp_wor[V[2]]
        end
    end
end

 function get_dist(prim::Primitives, res::Results, t::Int64)
    @unpack a_grid, length_a_grid, Π, Jr, N, T = prim
    psi_ret_new = Array{Float64}(zeros(length_a_grid, N-Jr+1))
    psi_wor_new = Array{Float64}(cat(zeros(length_a_grid, Jr-1), zeros(length_a_grid, Jr-1), dims=3))
    psi_wor_new[:,1,:] = res.psi_wor_tran[:,1,:,t]

    for j in 1:Jr-2
      for (i,k) in collect(Iterators.product(1:length_a_grid, 1:2))
        a_new = a_grid[i]
        psi_wor_new[i,j+1,k] = sum(res.psi_wor_tran[:,j,:,t].*(res.pol_func_wor_tran[:,j,:,t] .== a_new)* Π[:,k])
        end
    end
    res.psi_wor_tran[:,:,:,t+1] = psi_wor_new

    for i = 1:length_a_grid
        a_new = a_grid[i]
        psi_ret_new[i,1] = sum(res.psi_wor_tran[:,Jr-1,:,t].*(res.pol_func_wor_tran[:,Jr-1,:,t].== a_new))
    end

    for j in 1:N-Jr
      for i in 1:length_a_grid
        a_new = a_grid[i] # a'
        psi_ret_new[i,j+1] = sum(res.psi_ret_tran[:,j,t].*(res.pol_func_ret_tran[:,j,t] .== a_new))
        end
    end
    res.psi_ret_tran[:,:,t+1] = psi_ret_new
end

 function get_trans(prim::Primitives, res::Results, tol::Float64=1e-4, err::Float64=100.0)
    @unpack length_a_grid, T, mu, α, δ, Jr, N, ef = prim
    @unpack Zs, θ = res
    for i in 2:T+1
        res.psi_wor_tran[1,1,1,i] = sum(res.psi_wor_tran[:,1,1,i])
        res.psi_wor_tran[2:length_a_grid,1,1,i] .= 0
        res.psi_wor_tran[1,1,2,i] = sum(res.psi_wor_tran[:,1,2,i])
        res.psi_wor_tran[2:length_a_grid,1,2,i] .= 0
    end

    while err > tol
        Ks_old = res.Ks

        for t in T:-1:1
            θ = θ * (t < 1)
            res.b = θ * (1 - α) * res.Ks[t]^α * res.Ls[t]^(1-α) / sum(mu[Jr:N])
            res.r = α * res.Ks[t]^(α-1) * res.Ls[t]^(1-α) - δ
            res.w = (1-α) * res.Ks[t]^α * res.Ls[t]^(-α)
            println("=================================================")
            println("Policy Function Solving in period ", t)
            Bellman_ret(prim, res, t)
            Bellman_wor(prim, res, t)

            get_dist(prim, res, t)
            println("Policy Function Solved in period ", t)
            println("=================================================")
        end

        for t in 1:T
           println("=================================================")
           println("Distribution Solving in period ", t)
           get_dist(prim, res, t)
           println("Distribution Solved in period ", t)
           println("=================================================")
        end

        Ks_new = zeros(T+1)
        Ks_new[1] = Ks_old[1]

        Ls_new = zeros(T)
        for t in 1:T
            L_old = res.Ls[t]
            K_old = Ks_old[t+1]

            L_new = 0.99 * L_old + 0.01 * sum(((res.psi_wor_tran[:,:,1,t].*res.lab_func_wor_tran[:,:,1,t]) .* Zs[1] + (res.psi_wor_tran[:,:,2,t].*res.lab_func_wor_tran[:,:,2,t]) .* Zs[2]) .* repeat(ef, 1, length_a_grid)' .* repeat(mu[1:Jr-1], 1, length_a_grid)')
            K_new = 0.99 * K_old + 0.01 * (sum((res.psi_ret_tran[:,:,t] .* res.pol_func_ret_tran[:,:,t]) * mu[Jr:N]) + sum((res.psi_wor_tran[:,:,1,t].*res.pol_func_wor_tran[:,:,1,t]) * mu[1:Jr-1]) + sum((res.psi_wor_tran[:,:,2,t].*res.pol_func_wor_tran[:,:,2,t]) * mu[1:Jr-1]))
            Ls_new[t] = L_new
            Ks_new[t+1] = K_new
        end
        errL = maximum(abs.(Ls_new - Ls_old))
        errK = maximum(abs.(Ks_new - Ks_old))
        err = max(errL, errK)
        println("Aggregate Error: ", err)
        res.Ks = Ks_new
        res.Ls = LS_new
    end
end

 function Solve_trans(prim::Primitives, res::Results)
    get_trans(prim, res)
end
