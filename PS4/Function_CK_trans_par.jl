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
    a_max::Float64 = 75 #asset upper bound
    length_a_grid::Int64 = 5000 #number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a_min, length = length_a_grid, stop = a_max)) #asset grid
    Π::Matrix{Float64} = [0.9261 0.0739; 0.0189 0.9811] #transition matrix
    ef::Matrix{Float64} = DelimitedFiles.readdlm("/Users/Yeonggyu/Desktop/Econ 899 - Computation/PS/PS3/ef.txt", '\n')
    mu::Array{Float64} = cumprod([1; ones(N-1)./(1+n)])./sum(cumprod([1; ones(N-1)./(1+n)]))
end

#structure that holds model results
mutable struct Results
    # Stationary equilibrium
    val_func_ret::Array{Float64, 2} #value function retired
    pol_func_ret::Array{Float64, 2} #policy function retired
    val_func_wor::Array{Float64, 3} #value function workers
    pol_func_wor::Array{Float64, 3} #policy function workers - saving
    lab_func_wor::Array{Float64, 3} #labor supply function workers

    # Prices and aggregates
    r::Float64  #interest rate
    w::Float64  #wage
    L::Float64  #aggregate labor
    K::Float64  #aggregate capital
    b::Float64  #social security benefit

    # Policy experiments
    θ::Float64  #labor income tax
    γ::Float64  #weight on consumption
    Zs::Array{Float64} #idiosyncratic productivity

    # Aggregate capital transitions
    psi_ret::Array{Float64, 2}
    psi_wor::Array{Float64, 3}
end

#function for initializing model primitives and results
function Initialize(θ::Float64, Zs::Array{Float64, 1}, γ::Float64)
    prim = Primitives() #initialize primtiives
    val_func_ret = Array{Float64}(zeros(prim.length_a_grid, prim.N-prim.Jr+1)) #initial value function guess
    pol_func_ret = Array{Float64}(zeros(prim.length_a_grid, prim.N-prim.Jr+1)) #initial policy function guess
    val_func_wor = Array{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2))
    pol_func_wor = Array{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2))
    lab_func_wor = Array{Float64}(zeros(prim.length_a_grid, prim.Jr-1, 2))
    psi_ret = Array{Float64}(ones(prim.length_a_grid, prim.N-prim.Jr+1) ./ prim.length_a_grid)
    psi_wor = Array{Float64}(cat(reshape(ones(prim.length_a_grid), prim.length_a_grid, 1) * reshape(ones(prim.Jr-1)./prim.length_a_grid, 1, prim.Jr-1) .* 0.2037,
            reshape(ones(prim.length_a_grid), prim.length_a_grid, 1) * reshape(ones(prim.Jr-1)./prim.length_a_grid, 1, prim.Jr-1) .* 0.7963, dims=3)) #fraction of agents in each age group by state
    r::Float64 = 0.05 #interest rate
    w::Float64 = 1.05 #wage
    L::Float64 = 0.5
    K::Float64 = 3.2
    b::Float64 = 0.2
    res = Results(val_func_ret, pol_func_ret, val_func_wor, pol_func_wor, lab_func_wor, r, w, L, K, b, θ, γ, Zs, psi_ret, psi_wor) #initialize results struct
    prim, res #return deliverables
end

# T operator
function Bellman_ret(prim::Primitives,res::Results)
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
              if (1+r) * a + b > 0
                 ctmp_ret = (1+r) * a + b - a_grid[ap_index]
                 ctmp_ret = ifelse(ctmp_ret > 0, 1, 0)*ctmp_ret

                 vtmp_ret = ctmp_ret^((1-σ)*γ) /(1-σ) + β * res.val_func_ret[ap_index,j+1]
                    if vtmp_ret < val_up
                       next_val[a_index,j] = val_up
                       res.pol_func_ret[a_index,j] = a_grid[ap_index-1]
                       choice_lower = ap_index - 1
                       break
                    elseif ap_index == length_a_grid
                       next_val[a_index,j] = vtmp_ret
                       res.pol_func_ret[a_index,j] = a_grid[ap_index]
                    end
                 val_up = vtmp_ret
              elseif (1+r) * a + b <= 0
                 next_val[a_index,j] = -Inf
                 res.pol_func_ret[a_index,j] = a_grid[1]
              end
           end
       end
   end
   next_val
end

function Bellman_wor(prim::Primitives, res::Results)
   @unpack r, w, b, θ, γ, Zs = res #unpack value function
   @unpack a_grid, β, N, Jr, σ, length_a_grid, Π, ef = prim #unpack model primitives
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
            vtmp_wor = (ctmp_wor^γ * (1 -ltmp_wor)^(1-γ))^(1-σ) / (1-σ) + β * res.val_func_ret[ip,1]

            if vtmp_wor < val_up
               next_val[i,Jr-1,j] = val_up
               res.pol_func_wor[i,Jr-1,j] = a_grid[ip-1]
               res.lab_func_wor[i,Jr-1,j] = min(1, max((γ * (1 - θ) * ef[Jr-1] * Zs[j]* w - (1-γ)* ((1+r) * a - a_grid[ip-1])) / ((1-θ) * w * ef[Jr-1] * Zs[j]), 0))
               choice_lower = ip - 1
               break
            elseif ip == length_a_grid
               next_val[i,Jr-1,j] = vtmp_wor
               res.pol_func_wor[i,Jr-1,j] = a_grid[ip]
               res.lab_func_wor[i,Jr-1,j] = min(1, max((γ * (1 - θ) * ef[Jr-1] * Zs[j] * w - (1-γ) * ((1+r) * a - a_grid[ip])) / ((1-θ) * w * ef[Jr-1] * Zs[j]), 0))
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
              vtmp_wor = (ctmp_wor^γ * (1 - ltmp_wor)^(1-γ))^(1-σ) / (1-σ) + β * sum(res.val_func_wor[ip,j+1,:].* Π[k,:])

              if vtmp_wor < val_up
                 next_val[i,j,k] = val_up
                 res.pol_func_wor[i,j,k] = a_grid[ip-1]
                 res.lab_func_wor[i,j,k] = min(1, max((γ * (1 - θ) * ef[j] * Zs[k] * w - (1-γ) * ((1+r) * a - a_grid[ip-1])) / ((1-θ) * w * ef[j] * Zs[k]), 0))
                 choice_lower = ip - 1
                 break
              elseif ip == length_a_grid
                 next_val[i,j,k] = vtmp_wor
                 res.pol_func_wor[i,j,k] = a_grid[ip]
                 res.lab_func_wor[i,j,k] = min(1, max((γ * (1 - θ) * ef[j] * Zs[k] * w - (1-γ) * ((1+r) * a - a_grid[ip])) / ((1-θ) * w * ef[j] * Zs[k]), 0))
              end
              val_up = vtmp_wor
           end
       end
    end
  end
  next_val
end

function get_dist(prim::Primitives, res::Results)
   @unpack a_grid, length_a_grid, Π, Jr, N = prim
   psi_ret_new = Array{Float64}(zeros(length_a_grid, N-Jr+1))
   psi_wor_new = Array{Float64}(cat(zeros(length_a_grid, Jr-1), zeros(length_a_grid, Jr-1), dims=3))
   psi_wor_new[:,1,:] = res.psi_wor[:,1,:]

   for j in 1:Jr-2
       for (i,k) in collect(Iterators.product(1:length_a_grid, 1:2))
          if res.psi_wor[i,j,k] > 0
            i_p = argmax(a_grid .== res.pol_func_wor[i,j,k])

            for m in 1:2
               psi_wor_new[i_p,j+1,m] += res.psi_wor[i,j,k] * Π[k,m]
            end
          end
       end
   end
   res.psi_wor = psi_wor_new

   for i = 1:length_a_grid
       for k in 1:2
          if res.psi_wor[i,Jr-1,k] > 0
            i_p = argmax(a_grid .== res.pol_func_wor[i,Jr-1,k])

            psi_ret_new[i_p,1] += res.psi_wor[i,Jr-1,k]
         end
       end
   end

   for j in 1:N-Jr
       for i in 1:length_a_grid
          if res.psi_ret[i,j] > 0
            i_p = argmax(a_grid .== res.pol_func_ret[i,j])
            psi_ret_new[i_p,j+1] += res.psi_ret[i,j]
         end
       end
   end
   res.psi_ret = psi_ret_new
end

function get_prices(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    @unpack α, δ, a_grid, length_a_grid, Π, Jr, N, ef, mu = prim
    @unpack Zs, θ = res
    ED = 0.01 # excess demand
    n = 0

    res.psi_wor[1, 1, 1] = sum(res.psi_wor[:,1,1])
    res.psi_wor[1, 1, 2] = sum(res.psi_wor[:,1,2])
    res.psi_wor[2:length_a_grid, 1,:] .= 0

    while abs(ED) > tol
        println("**************************************************")
        println(n+1, "th Iteration \n", "SOLVING DISTRIBUTION PROBLEM")

        res.b = θ * (1 - α) * res.K^α * res.L^(1-α) / sum(mu[Jr:N])
        res.r = α * res.K^(α-1) * res.L^(1-α) - δ
        res.w = (1-α) * res.K^α * res.L^(-α)

        println("SOLVING HOUSEHOLD PROBLEM \n")

        next_val_ret = Bellman_ret(prim, res)
        next_val_wor = Bellman_wor(prim, res)
        res.val_func_ret = next_val_ret
        res.val_func_wor = next_val_wor

        println("HOUSEHOLD PROBLEM SOLVED \n")

        get_dist(prim, res)

        println("DISTRIBUTION UPDATED \n")

        K_old = res.K
        K_new = 0.99 * K_old + 0.01 * (sum((res.psi_ret .* res.pol_func_ret) * mu[Jr:N]) + sum((res.psi_wor[:,:,1].*res.pol_func_wor[:,:,1]) * mu[1:Jr-1]) + sum((res.psi_wor[:,:,2].*res.pol_func_wor[:,:,2]) * mu[1:Jr-1]))
        errK = abs(K_new - K_old)
        res.K = K_new

        L_old = res.L
        L_new = 0.99 * L_old + 0.01 * sum(((res.psi_wor[:,:,1].*res.lab_func_wor[:,:,1]) .* Zs[1] + (res.psi_wor[:,:,2].*res.lab_func_wor[:,:,2]) .* Zs[2]) .* repeat(ef, 1, length_a_grid)' .* repeat(mu[1:Jr-1], 1, length_a_grid)')
        errL = abs(L_new - L_old)
        res.L = L_new

        println("Capital is now ", K_new, " and Labor is now ", L_new, "\n")
        println("**************************************************")

        ED = maximum([errK, errL])
        n+=1
    end
    println("\n******************************************************************\n")
    @printf "Aggregate Error = %.6f in iteration %.6f is within threshold!\n\n" ED n
    println("******************************************************************\n")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    get_prices(prim, res)
end
