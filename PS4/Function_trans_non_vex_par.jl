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
end

 mutable struct Results
    # Transition path
    val_func_ret_tran::Array{Float64, 2} #value function retired
    pol_func_ret_tran::Array{Float64, 3} #policy function retired
    val_func_wor_tran::Array{Float64, 3} #value function workers
    pol_func_wor_tran::Array{Float64, 4} #policy function workers - saving
    lab_func_wor_tran::Array{Float64, 4} #labor supply function workers

    val_ter_ret::Array{Float64, 2} #terminal value retired
    val_ter_wor::Array{Float64, 3} #terminal value worker

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
    T::Int64 # total periods
    NT::Int64 # number of transition periods
end

#function for initializing model primitives and results
 function Initialize(θ::Float64, Zs::Array{Float64, 1}, γ::Float64, K1::Float64, K2::Float64, Vwor::Array{Float64, 3}, Vret::Array{Float64, 2}, Psiwor::Array{Float64, 3}, Psiret::Array{Float64, 2}, T::Int64, NT::Int64)
    prim = Primitives() #initialize primtiives

    val_func_ret_tran = Array{Float64}(zeros(prim.length_a_grid, prim.N-prim.Jr+1))  #value function retired
    pol_func_ret_tran = Array{Float64}(zeros(prim.length_a_grid, prim.N-prim.Jr+1, T))  #policy function retired
    val_func_wor_tran = Array{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2)) #value function workers
    pol_func_wor_tran = Array{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2, T)) #policy function workers - saving
    lab_func_wor_tran = Array{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2, T)) #labor supply function workers

    val_func_ret_tran = Vret
    val_func_wor_tran = Vwor

    val_ter_ret = Vret
    val_ter_wor = Vwor

    psi_ret_tran = Array{Float64}(ones(prim.length_a_grid, prim.N-prim.Jr+1, T+1) ./ prim.length_a_grid)
    psi_wor_tran = Array{Float64}(ones(prim.length_a_grid, prim.Jr-1, 2, T+1) ./ prim.length_a_grid) #fraction of agents in each age group by state

    psi_wor_tran[:,:,1,:] = psi_wor_tran[:,:,1,:] .* 0.2037
    psi_wor_tran[:,:,2,:] = psi_wor_tran[:,:,2,:] .* 0.7963
    psi_ret_tran[:,:,1] = Psiret
    psi_wor_tran[:,:,:,1] = Psiwor

    r::Float64 = 0.05 #interest rate
    w::Float64 = 0.99 #wage
    b::Float64 = 0.2
    Ls::Array{Float64} = ones(T) .* 0.35
    Ks::Array{Float64} = collect(range(K1, K2, length=T+1))
    res = Results(val_func_ret_tran, pol_func_ret_tran, val_func_wor_tran, pol_func_wor_tran, lab_func_wor_tran, val_ter_ret, val_ter_wor, r, w, b, θ, γ, Zs, Ls, Ks, psi_ret_tran, psi_wor_tran, T, NT) #initialize results struct
    prim, res #return deliverables
end

# T operator
 function Bellman_ret(prim::Primitives,res::Results, t::Int64)
    @unpack r, γ, b = res #unpack value function
    @unpack a_grid, β, N, Jr, σ, length_a_grid = prim #unpack model primitives
    ctmp_ter = (1+r) * a_grid .+ b
    next_val = Array{Float64}(zeros(length_a_grid, N-Jr+1))
    next_val[:,N-Jr+1] = ctmp_ter.^((1-σ)*γ) ./ (1 - σ)

    for j in N-Jr:-1:1 # retirement group iteration
      choice_lower = 1
        for a_index = 1:length_a_grid
         a = a_grid[a_index] #value of k
         val_up = -Inf
            for ap_index = choice_lower:length_a_grid
            ctmp_ret = (1+r) * a + b - a_grid[ap_index]
            ctmp_ret = ifelse(ctmp_ret > 0, 1, 0)*ctmp_ret

            vtmp_ret = ctmp_ret^((1-σ)*γ) /(1-σ) + β * res.val_func_ret_tran[ap_index,j+1]
               if vtmp_ret < val_up
                  next_val[a_index,j] = val_up
                  res.pol_func_ret_tran[a_index,j,t] = a_grid[ap_index-1]
                  choice_lower = ap_index - 1
                  break
               elseif ap_index == length_a_grid
                  next_val[a_index,j] = vtmp_ret
                  res.pol_func_ret_tran[a_index,j,t] = a_grid[ap_index]
               end
            val_up = vtmp_ret
            end
        end
    end
    next_val
end

function Bellman_wor(prim::Primitives, res::Results, t::Int64)
    @unpack r, w, b, θ, γ, Zs, T, NT = res #unpack value function
    @unpack a_grid, β, N, Jr, σ, length_a_grid, Π, ef = prim #unpack model primitives
    θ = θ * ifelse(t <= T-NT+1, 1, 0)
    next_val = Array{Float64}(zeros(length_a_grid, Jr-1, 2))

    for j = 1:2
        choice_lower = 1
          for i = 1:length_a_grid
           a = a_grid[i]
           val_up = -Inf
           for ip = choice_lower:length_a_grid
             ltmp_wor = min(1, max((γ * (1 - θ) * ef[Jr-1] * Zs[j] * w - (1-γ) * ((1+r) * a - a_grid[ip])) / ((1-θ) * w * ef[Jr-1] * Zs[j]), 0))
             ctmp_wor = w * (1 - θ) * ef[Jr-1] * Zs[j] * ltmp_wor + (1 + r) * a - a_grid[ip]
             ctmp_wor = ifelse(ctmp_wor > 0, 1, 0) * ctmp_wor
             vtmp_wor = (ctmp_wor^γ * (1 -ltmp_wor)^(1-γ))^(1-σ) / (1-σ) + β * res.val_func_ret_tran[ip,1]

             if vtmp_wor < val_up
                next_val[i,Jr-1,j] = val_up
                res.pol_func_wor_tran[i,Jr-1,j,t] = a_grid[ip-1]
                res.lab_func_wor_tran[i,Jr-1,j,t] = min(1, max((γ * (1 - θ) * ef[Jr-1] * Zs[j]* w - (1-γ)* ((1+r) * a - a_grid[ip-1])) / ((1-θ) * w * ef[Jr-1] * Zs[j]), 0))
                choice_lower = ip - 1
                break
             elseif ip == length_a_grid
                next_val[i,Jr-1,j] = vtmp_wor
                res.pol_func_wor_tran[i,Jr-1,j,t] = a_grid[ip]
                res.lab_func_wor_tran[i,Jr-1,j,t] = min(1, max((γ * (1 - θ) * ef[Jr-1] * Zs[j] * w - (1-γ) * ((1+r) * a - a_grid[ip])) / ((1-θ) * w * ef[Jr-1] * Zs[j]), 0))
             end
             val_up = vtmp_wor
          end
       end
    end

    for j in Jr-2:-1:1
      for k = 1:2
         choice_lower = 1
           for i = 1:length_a_grid
            a = a_grid[i]
            val_up = -Inf
            for ip = choice_lower:length_a_grid
               ltmp_wor= min(1, max((γ * (1 - θ) * ef[j] * Zs[k] * w - (1-γ) * ((1+r) * a - a_grid[ip])) / ((1-θ) * w * ef[j] * Zs[k]), 0))
               ctmp_wor = w * (1 - θ) * ef[j] * Zs[k] * ltmp_wor + (1 + r) * a - a_grid[ip]
               ctmp_wor = ifelse(ctmp_wor> 0, 1, 0)*ctmp_wor
               vtmp_wor = (ctmp_wor^γ * (1 - ltmp_wor)^(1-γ))^(1-σ) / (1-σ) + β * sum(res.val_func_wor_tran[ip,j+1,:].* Π[k,:])

               if vtmp_wor < val_up
                  next_val[i,j,k] = val_up
                  res.pol_func_wor_tran[i,j,k,t] = a_grid[ip-1]
                  res.lab_func_wor_tran[i,j,k,t] = min(1, max((γ * (1 - θ) * ef[j] * Zs[k] * w - (1-γ) * ((1+r) * a - a_grid[ip-1])) / ((1-θ) * w * ef[j] * Zs[k]), 0))
                  choice_lower = ip - 1
                  break
               elseif ip == length_a_grid
                  next_val[i,j,k] = vtmp_wor
                  res.pol_func_wor_tran[i,j,k,t] = a_grid[ip]
                  res.lab_func_wor_tran[i,j,k,t] = min(1, max((γ * (1 - θ) * ef[j] * Zs[k] * w - (1-γ) * ((1+r) * a - a_grid[ip])) / ((1-θ) * w * ef[j] * Zs[k]), 0))
               end
               val_up = vtmp_wor
            end
        end
     end
   end
   next_val
end

 function get_dist(prim::Primitives, res::Results, t::Int64)
    @unpack a_grid, length_a_grid, Π, Jr, N = prim
    @unpack T, NT = res
    psi_ret_new = Array{Float64}(zeros(length_a_grid, N-Jr+1))
    psi_wor_new = Array{Float64}(cat(zeros(length_a_grid, Jr-1), zeros(length_a_grid, Jr-1), dims=3))
    psi_wor_new[:,1,:] = res.psi_wor_tran[:,1,:,t]

    for j in 1:Jr-2
        for (i,k) in collect(Iterators.product(1:length_a_grid, 1:2))
           if res.psi_wor_tran[i,j,k,t] > 0
             i_p = argmax(a_grid .== res.pol_func_wor_tran[i,j,k,t])

             for m in 1:2
                psi_wor_new[i_p,j+1,m] += res.psi_wor_tran[i,j,k,t] * Π[k,m]
             end
           end
        end
    end
    res.psi_wor_tran[:,:,:,t+1] = psi_wor_new

    for i = 1:length_a_grid
        for k in 1:2
           if res.psi_wor_tran[i,Jr-1,k,t] > 0
             i_p = argmax(a_grid .== res.pol_func_wor_tran[i,Jr-1,k,t])

             psi_ret_new[i_p,1] += res.psi_wor_tran[i,Jr-1,k,t]
          end
        end
    end

    for j in 1:N-Jr
        for i in 1:length_a_grid
           if res.psi_ret_tran[i,j,t] > 0
             i_p = argmax(a_grid .== res.pol_func_ret_tran[i,j,t])
             psi_ret_new[i_p,j+1] += res.psi_ret_tran[i,j,t]
          end
        end
    end
    res.psi_ret_tran[:,:,t+1] = psi_ret_new
end

 function get_trans(prim::Primitives, res::Results, tol::Float64=1e-4, err::Float64=100.0)
    @unpack length_a_grid, mu, α, δ, Jr, N, ef = prim
    @unpack Zs, θ, T, NT = res
    for i in 2:T+1
        res.psi_wor_tran[1,1,1,i] = sum(res.psi_wor_tran[:,1,1,i])
        res.psi_wor_tran[2:length_a_grid,1,1,i] .= 0
        res.psi_wor_tran[1,1,2,i] = sum(res.psi_wor_tran[:,1,2,i])
        res.psi_wor_tran[2:length_a_grid,1,2,i] .= 0
    end

    counter = 0
    while err > tol
        counter+=1
        Ks_old = res.Ks
        Ls_old = res.Ls
        println("Iteration: ", counter)
        println("=================================================")
        println("Policy Function Solving")

        res.val_func_ret_tran = res.val_ter_ret # Put V_T as the terminal value function to begin the iteration backwards
        res.val_func_wor_tran = res.val_ter_wor # Put V_T as the terminal value function to begin the iteration backwards
        for t in T:-1:1
            res.b = (θ * (1 - α) * res.Ks[t]^α * res.Ls[t]^(1-α) / sum(mu[Jr:N])) * ifelse(t <= T-NT+1, 1, 0)
            res.r = α * res.Ks[t]^(α-1) * res.Ls[t]^(1-α) - δ
            res.w = (1-α) * res.Ks[t]^α * res.Ls[t]^(-α)
            next_val_ret = Bellman_ret(prim, res, t)
            next_val_wor = Bellman_wor(prim, res, t)
            res.val_func_ret_tran = next_val_ret
            res.val_func_wor_tran = next_val_wor
        end
        println("Policy Function Solved")
        println("=================================================")

        println("=================================================")
        println("Distribution Solving")
        for t in 1:T
           get_dist(prim, res, t)
        end
        println("Distribution Solved")
        println("=================================================")

        Ks_new = Array{Float64}(zeros(T+1))
        Ks_new[1] = Ks_old[1]

        Ls_new = Array{Float64}(zeros(T))
          for t in 1:T
            L_new = sum(((res.psi_wor_tran[:,:,1,t].*res.lab_func_wor_tran[:,:,1,t]) .* Zs[1] + (res.psi_wor_tran[:,:,2,t].*res.lab_func_wor_tran[:,:,2,t]) .* Zs[2]) .* repeat(ef, 1, length_a_grid)' .* repeat(mu[1:Jr-1], 1, length_a_grid)')
            K_new = sum((res.psi_ret_tran[:,:,t] .* res.pol_func_ret_tran[:,:,t]) * mu[Jr:N]) + sum((res.psi_wor_tran[:,:,1,t].*res.pol_func_wor_tran[:,:,1,t]) * mu[1:Jr-1]) + sum((res.psi_wor_tran[:,:,2,t].*res.pol_func_wor_tran[:,:,2,t]) * mu[1:Jr-1])

            Ls_new[t] = L_new
            Ks_new[t+1] = K_new
        end
        errK = maximum(abs.(Ks_new - Ks_old))
        errL = maximum(abs.(Ls_new - Ls_old))
        err = errK + errL
        println("Aggregate Error: ", err, " in Iteration ", counter)
        res.Ks = 0.2 .* Ks_new + 0.8 .* Ks_old
        res.Ls = 0.2 .* Ls_new + 0.8 .* Ls_old
    end
end

 function Solve_trans(prim::Primitives, res::Results)
    get_trans(prim, res)
end
