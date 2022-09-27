#parameters
@everywhere @with_kw struct Primitives
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
    ef::Matrix{Float64} = DelimitedFiles.readdlm("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS3/ef.txt", '\n')
    mu::Array{Float64} = cumprod([1; ones(N-1)./(1+n)])./sum(cumprod([1; ones(N-1)./(1+n)]))
end

#structure that holds model results
@everywhere mutable struct Results
    val_func_ret::SharedArray{Float64, 2} #value function retired
    pol_func_ret::SharedArray{Float64, 2} #policy function retired
    val_func_wor::SharedArray{Float64, 3} #value function workers
    pol_func_wor::SharedArray{Float64, 3} #policy function workers - saving
    lab_func_wor::SharedArray{Float64, 3} #labor supply function workers
    r::Float64  #interest rate
    w::Float64  #wage
    L::Float64  #aggregate labor
    K::Float64  #aggregate capital
    b::Float64  #social security benefit
    θ::Float64  #labor income tax
    γ::Float64  #weight on consumption
    Zs::Array{Float64} #idiosyncratic productivity
    psi_ret::SharedArray{Float64, 2}
    psi_wor::SharedArray{Float64, 3}
end

#function for initializing model primitives and results
@everywhere function Initialize(θ::Float64, Zs::Array{Float64, 1}, γ::Float64)
    prim = Primitives() #initialize primtiives
    val_func_ret = SharedArray{Float64}(zeros(prim.length_a_grid, prim.N-prim.Jr+1)) #initial value function guess
    pol_func_ret = SharedArray{Float64}(zeros(prim.length_a_grid, prim.N-prim.Jr+1)) #initial policy function guess
    val_func_wor = SharedArray{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2))
    pol_func_wor = SharedArray{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2))
    lab_func_wor = SharedArray{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2))
    psi_ret = SharedArray{Float64}(ones(prim.length_a_grid, prim.N-prim.Jr+1) ./ prim.length_a_grid)
    psi_wor = SharedArray{Float64}(cat(reshape(ones(prim.length_a_grid), prim.length_a_grid, 1) * reshape(ones(prim.Jr-1)./prim.length_a_grid, 1, prim.Jr-1) .* 0.2037,
            reshape(ones(prim.length_a_grid), prim.length_a_grid, 1) * reshape(ones(prim.Jr-1)./prim.length_a_grid, 1, prim.Jr-1) .* 0.7963, dims=3)) #fraction of agents in each age group by state
    r::Float64 = 0.05 #interest rate
    w::Float64 = 1.05 #wage
    L::Float64 = 1
    K::Float64 = 3.2
    b::Float64 = 0.2
    res = Results(val_func_ret, pol_func_ret, val_func_wor, pol_func_wor, lab_func_wor, r, w, L, K, b, θ, γ, Zs, psi_ret, psi_wor) #initialize results struct
    prim, res #return deliverables
end

# T operator
@everywhere function Bellman(prim::Primitives,res::Results)
    @unpack val_func_ret, val_func_wor, r, w, b θ, γ, Zs = res #unpack value function
    @unpack a_grid, β, n, N, Jr, σ, length_a_grid, Π, ef = prim #unpack model primitives
    ctmp_ret = SharedArray{Float64}(zeros(length_a_grid)) #consumption matrix to fill
    vtmp_ret = SharedArray{Float64}(zeros(length_a_grid))

    ctmp_ret = (1+r) .* a_grid .+ b
    val_func_ret[:,N-Jr+1] = ctmp_ret.^((1-σ)*γ) ./ (1 - σ)

    for j in N-Jr:-1:1 # retirement group iteration
        @sync @distributed for a_index = 1:length_a_grid
            a = a_grid[a_index] #value of k
            ctmp_ret = (1+r) * a .+ b .- a_grid
            ctmp_ret = ifelse.(ctmp_ret .> 0, 1, 0).*ctmp_ret

            vtmp_ret = ctmp_ret.^((1-σ)*γ) ./(1-σ) .+ β .* res.val_func_ret[:,j+1]
            V = findmax(vtmp_ret)

            res.val_func_ret[a_index,j] = V[1]
            res.pol_func_ret[a_index,j] = a_grid[V[2]]
        end
    end

    ctmp_wor = SharedArray{Float64}(zeros(length_a_grid))
    vtmp_wor = SharedArray{Float64}(zeros(length_a_grid))
    ltmp_wor = SharedArray{Float64}(zeros(length_a_grid))

    @sync @distributed for (i, j) in collect(Iterators.product(1:length_a_grid, 1:2))
        ltmp_wor = (γ .* (1 - θ) * ef[Jr-1] .* Zs[j] .* w .- (1-γ) .* ((1+r) .* a_grid[i] .- a_grid)) ./ ((1-θ) * w * ef[Jr-1] * Zs[j])
        ltmp_wor[ltmp_wor.< 0] .= 0
        ltmp_wor[ltmp_wor.> 1] .= 1

        ctmp_wor = w .* (1 - θ) .* ef[Jr-1] .* Zs[j] .* ltmp_wor .+ (1 + r) .* a_grid[i] .- a_grid
        ctmp_wor = ifelse.(ctmp_wor .> 0, 1, 0).*ctmp_wor
        vtmp_wor = (ctmp_wor.^γ .* (1 .-ltmp_wor).^(1-γ)).^(1-σ) ./ (1-σ) .+ β .* res.val_func_ret[:,1]

        V = findmax(vtmp_wor)

        res.val_func_wor[i,Jr-1,j] = V[1]
        res.pol_func_wor[i,Jr-1,j] = a_grid[V[2]]
        res.lab_func_wor[i,Jr-1,j] = ltmp_wor[V[2]]
    end

    for j in Jr-2:-1:1
        @sync @distributed for (i,k) in collect(Iterators.product(1:length_a_grid, 1:2))
        a = a_grid[i]
        ltmp_wor= (γ .* (1 - θ) * ef[j] .* Zs[k] .* w .- (1-γ) .* ((1+r) .* a .- a_grid)) ./ ((1-θ) * w * ef[j] * Zs[k])
        ltmp_wor[ltmp_wor .< 0] .= 0
        ltmp_wor[ltmp_wor .> 1] .= 1

        ctmp_wor = w .* (1 - θ) .* ef[j] .* Zs[k] .* ltmp_wor .+ (1 + r) .* a .- a_grid
        ctmp_wor = ifelse.(ctmp_wor.> 0, 1, 0).*ctmp_wor

        vtmp_wor = (ctmp_wor.^γ .* (1 .- ltmp_wor).^(1-γ)).^(1-σ) ./ (1-σ) .+ β .* res.val_func_wor[:,j+1,:] * Π[k,:]

        V = findmax(vtmp_wor)
        res.val_func_wor[i,j,k] = V[1]
        res.pol_func_wor[i,j,k] = a_grid[V[2]]
        res.lab_func_wor[i,j,k] = ltmp_wor[V[2]]
        end
    end
end

@everywhere function get_psi(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    @unpack α, δ, a_grid, length_a_grid, Π, Jr, N, ef, mu = prim
    @unpack Zs, θ = res
    ED = 0.01 # excess demand
    n = 0

    res.psi_wor[1, 1, 1] = sum(res.psi_wor[:,1,1])
    res.psi_wor[1, 1, 2] = sum(res.psi_wor[:,1,2])
    res.psi_wor[2:length_a_grid, 1,:] .= 0

    while abs(ED) > tol
        println("SOLVING DISTRIBUTION PROBLEM \n", n, "th Iteration")

        res.b = θ * (1 - α) * res.K^α * res.L^(1-α) / sum(mu[Jr:N])
        res.r = α * res.K^(α-1) * res.L^(1-α) - δ
        res.w = (1-α) * res.K^α * res.L^(-α)

        println("SOLVING HOUSEHOLD PROBLEM \n")

        Bellman(prim, res)

        println("HOUSEHOLD PROBLEM SOLVED \n")

        psi_ret_new = SharedArray{Float64}(zeros(length_a_grid, N-Jr+1))
        psi_wor_new = SharedArray{Float64}(cat(zeros(length_a_grid, Jr-1), zeros(length_a_grid, Jr-1), dims=3))
        psi_wor_new[:,1,:] = res.psi_wor[:,1,:]

        for j in 1:Jr-2
            @sync @distributed for (i,k) in collect(Iterators.product(1:length_a_grid, 1:2))
            a_new = a_grid[i]
            psi_wor_new[i,j+1,k] = sum(res.psi_wor[:,j,:].*(res.pol_func_wor[:,j,:] .== a_new)* Π[:,k])
            end
        end
        res.psi_wor = psi_wor_new

        @sync @distributed for i = 1:length_a_grid
            a_new = a_grid[i]
            psi_ret_new[i,1] = sum(res.psi_wor[:,Jr-1,:].*(res.pol_func_wor[:,Jr-1,:].== a_new))
        end

        for j in 1:N-Jr
            @sync @distributed for i in 1:length_a_grid
            a_new = a_grid[i] # a'
            psi_ret_new[i,j+1] = sum(res.psi_ret[:,j].*(res.pol_func_ret[:,j] .== a_new))
            end
        end

        res.psi_ret = psi_ret_new
        println("DISTRIBUTION PROBLEM SOLVED \n")

        K_old = res.K
        K_new = 0.99 * K_old + 0.01 * (sum((res.psi_ret .* res.pol_func_ret) * mu[Jr:N]) + sum((res.psi_wor[:,:,1].*res.pol_func_wor[:,:,1]) * mu[1:Jr-1]) + sum((res.psi_wor[:,:,2].*res.pol_func_wor[:,:,2]) * mu[1:Jr-1]))
        errK = abs(K_new - K_old)
        res.K = K_new

        L_old = res.L
        L_new = 0.99 * L_old + 0.01 * sum(((res.psi_wor[:,:,1].*res.lab_func_wor[:,:,1]) .* Zs[1] + (res.psi_wor[:,:,2].*res.lab_func_wor[:,:,2]) .* Zs[2]) .* repeat(ef, 1, length_a_grid)' .* repeat(mu[1:Jr-1], 1, length_a_grid)')
        errL = abs(L_new - L_old)
        res.L = L_new

        println("Capital is now ", K_new, " and Labor is now ", L_new, "\n")

        ED = maximum([errK, errL])
        n+=1
    end
    println("\n******************************************************************\n")
    @printf "Aggregate Error = %.6f in iteration %.6f is within threshold!\n\n" ED n
    println("******************************************************************\n")
end

#solve the model
@everywhere function Solve_model(prim::Primitives, res::Results)
    get_psi(prim, res)
end
