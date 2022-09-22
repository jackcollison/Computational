#Runs slower than single-core on Macbook M1 Pro...why?

#parameters
@everywhere @with_kw struct Primitives
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
    a_max::Float64 = 50 #asset upper bound
    length_a_grid::Int64 = 500 #number of asset grid points
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
    psi_ret::SharedArray{Float64, 2}
    psi_wor::SharedArray{Float64, 3}
end

#function for initializing model primitives and results
@everywhere function Initialize()
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
    K::Float64 = 3
    b::Float64 = 0.2
    res = Results(val_func_ret, pol_func_ret, val_func_wor, pol_func_wor, lab_func_wor, r, w, L, K, b, psi_ret, psi_wor) #initialize results struct
    prim, res #return deliverables
end

# T operator
@everywhere function Bellman(prim::Primitives,res::Results)
    @unpack val_func_ret, val_func_wor, r, w, b = res #unpack value function
    @unpack a_grid, β, n, N, Jr, θ, σ, γ, length_a_grid, Π, Zs, ef = prim #unpack model primitives
    v_next_ret = SharedArray{Float64}(zeros(length_a_grid, N-Jr+1)) #next guess of value function to fill
    ctmp_ret = SharedArray{Float64}(zeros(length_a_grid)) #consumption matrix to fill
    vtmp_ret = SharedArray{Float64}(zeros(length_a_grid))

    ctmp_ret = (1+r) .* a_grid .+ b
    val_func_ret[:,N-Jr+1] = ctmp_ret.^((1-σ)*γ) ./ (1 - σ)
    v_next_ret[:,N-Jr+1] = val_func_ret[:,N-Jr+1]

    @sync @distributed for (a_index, j) in collect(Iterators.product(1:length_a_grid, collect(N-Jr:-1:1))) # retirement group iteration
        a = a_grid[a_index] #value of k
        ctmp_ret = (1+r) * a .+ b .- a_grid
        ctmp_ret = ifelse.(ctmp_ret .> 0, 1, 0).*ctmp_ret

        vtmp_ret = ctmp_ret.^((1-σ)*γ) ./(1-σ) .+ β .* val_func_ret[:,j+1]
        v_next_ret[a_index,j] = maximum(vtmp_ret)
        res.pol_func_ret[a_index,j] = a_grid[findmax(vtmp_ret)[2]]
    end

    v_next_wor = SharedArray{Float64}(zeros(length_a_grid, Jr-1, 2))

    ctmp_wor = SharedArray{Float64}(zeros(length_a_grid))
    vtmp_wor = SharedArray{Float64}(zeros(length_a_grid))
    ltmp_wor = SharedArray{Float64}(zeros(length_a_grid))

    @sync @distributed for (i, j) in collect(Iterators.product(1:length_a_grid, 1:2))
        ltmp_wor = (γ .* (1 - θ) * ef[Jr-1] .* Zs[j] .* w .- (1-γ) .* ((1+r) .* a_grid[i] .- a_grid)) ./ ((1-θ) * w * ef[Jr-1] * Zs[j])
        ltmp_wor[ltmp_wor.< 0] .= 0
        ltmp_wor[ltmp_wor.> 1] .= 1

        ctmp_wor = w .* (1 - θ) .* ef[Jr-1] .* Zs[j] .* ltmp_wor .+ (1 + r) .* a_grid[i] .- a_grid
        ctmp_wor = ifelse.(ctmp_wor .> 0, 1, 0).*ctmp_wor
        vtmp_wor = (ctmp_wor.^γ .* (1 .-ltmp_wor).^(1-γ)).^(1-σ) ./ (1-σ) .+ β .* v_next_ret[:,1]

        v_next_wor[i,Jr-1,j] = maximum(vtmp_wor)
        res.pol_func_wor[i,Jr-1,j] = a_grid[findmax(vtmp_wor)[2]]
        res.lab_func_wor[i,Jr-1,j] = ltmp_wor[findmax(vtmp_wor)[2]]
    end

    val_func_wor[:,Jr-1,:] = v_next_wor[:,Jr-1,:]

    @sync @distributed for (i,j,k) in collect(Iterators.product(1:length_a_grid, collect(Jr-2:-1:1), 1:2))
        a = a_grid[i]
        ltmp_wor= (γ .* (1 - θ) * ef[j] .* Zs[k] .* w .- (1-γ) .* ((1+r) .* a .- a_grid)) ./ ((1-θ) * w * ef[j] * Zs[k])
        ltmp_wor[ltmp_wor .< 0] .= 0
        ltmp_wor[ltmp_wor .> 1] .= 1

        ctmp_wor = w .* (1 - θ) .* ef[j] .* Zs[k] .* ltmp_wor .+ (1 + r) .* a .- a_grid
        ctmp_wor = ifelse.(ctmp_wor.> 0, 1, 0).*ctmp_wor

        vtmp_wor = (ctmp_wor.^γ .* (1 .- ltmp_wor).^(1-γ)).^(1-σ) ./ (1-σ) .+ β .* val_func_wor[:,j+1,:] * Π[k,:]

        v_next_wor[i,j,k] = maximum(vtmp_wor)
        res.pol_func_wor[i,j,k] = a_grid[findmax(vtmp_wor)[2]]
        res.lab_func_wor[i,j,k] = ltmp_wor[findmax(vtmp_wor)[2]]
    end
    return v_next_wor, v_next_ret #return next guess of value function
end

#Value function iteration
@everywhere function get_g(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
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

@everywhere function get_psi(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    @unpack α, δ, θ, a_grid, length_a_grid, Π, Jr, N, ef, Zs, mu = prim
    ED = 0.01 # excess demand

        println("###############################################")
        println("######## SOLVING DISTRIBUTION PROBLEM #########")
        println("###############################################\n")

    res.psi_wor[1, 1, 1] = sum(res.psi_wor[1,:,1])
    res.psi_wor[1, 1, 2] = sum(res.psi_wor[1,:,2])
    res.psi_wor[2:length_a_grid, 1,:] .= 0

    while abs(ED) > tol
        err = 100.0

        res.b = θ * (1 - α) * res.K^α * res.L^(1-α) / sum(res.psi_ret * reshape(mu[Jr:N], N-Jr+1, 1))
        get_g(prim, res)

        while err>tol
            psi_ret_new = SharedArray{Float64}(zeros(length_a_grid, N-Jr+1))
            psi_wor_new = SharedArray{Float64}(cat(zeros(length_a_grid, Jr-1), zeros(length_a_grid, Jr-1), dims=3))
            psi_wor_new[:,1,:] = res.psi_wor[:,1,:]

            @sync @distributed for (i,j,k) in collect(Iterators.product(1:length_a_grid, 1:Jr-2, 1:2))
                a_new = a_grid[i]
                psi_wor_new[i,j+1,k] = sum(reshape(res.psi_wor[:,j,:].*(res.pol_func_wor[:,j,:] .== a_new), length_a_grid, 2) * Π[:,k])
            end
            err1 = maximum(abs.(psi_wor_new .- res.psi_wor)./abs.(res.psi_wor))
            res.psi_wor = psi_wor_new

            @sync @distributed for i = 1:length_a_grid
                a_new = a_grid[i]
                psi_ret_new[i,1] = sum(res.psi_wor[:,Jr-1,:].*(res.pol_func_wor[:,Jr-1,:].== a_new))
            end

            @sync @distributed for (i, j) in collect(Iterators.product(1:length_a_grid, 1:N-Jr)
                a_new = a_grid[i] # a'
                psi_ret_new[i,j+1] = sum(res.psi_ret[:,j].*(res.pol_func_ret[:,j] .== a_new))
            end

            err2 = maximum(abs.(psi_ret_new .- res.psi_ret)./abs.(res.psi_ret))
            res.psi_ret = psi_ret_new
            err = maximum([err1, err2])
        end

        K_old = res.K
        K_new = 0.99 * K_old + 0.01 * (sum((res.psi_ret .* res.pol_func_ret) * mu[Jr:N]) + sum((res.psi_wor[:,:,1].*res.pol_func_wor[:,:,1]) * mu[1:Jr-1]) + sum((res.psi_wor[:,:,2].*res.pol_func_wor[:,:,2]) * mu[1:Jr-1]))
        errK = abs(K_new - K_old)/K_old
        res.K = K_new

        L_old = res.L
        L_new = 0.99 * L_old + 0.01 * sum(((res.psi_wor[:,:,1].*res.lab_func_wor[:,:,1]) .* Zs[1] + (res.psi_wor[:,:,2].*res.lab_func_wor[:,:,2]) .* Zs[2]) .* repeat(ef, 1, length_a_grid)' .* repeat(mu[1:Jr-1], 1, length_a_grid)')
        errL = abs(L_new - L_old)/L_old
        res.L = L_new

        res.r = α * res.K^(α-1) * res.L^(1-α) - δ
        res.w = (1-α) * res.K^α * res.L^(-α)
        ED = maximum([errK, errL])
    end
    println("\n******************************************************************\n")
    @printf "Aggregate Error = %.6f is within threshold!\n\n" ED
    println("******************************************************************\n")
end

#solve the model
@everywhere function Solve_model(prim::Primitives, res::Results)
    get_psi(prim, res)
end
