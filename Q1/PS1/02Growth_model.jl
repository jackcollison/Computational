using Distributed
@everywhere using LinearAlgebra, SharedArrays

#keyword-enabled structure to hold model primitives
@everywhere @with_kw struct Primitives
    β::Float64 = 0.99 #discount rate
    δ::Float64 = 0.025 #depreciation rate
    α::Float64 = 0.36 #capital share
    k_min::Float64 = 0.01 #capital lower bound
    k_max::Float64 = 45.0 #capital upper bound
    nk::Int64 = 1000 #number of capital grid points
    k_grid::Array{Float64,1} = collect(range(k_min, length = nk, stop = k_max)) #capital grid
    markov::Array{Float64,2} = [0.977 0.023; 0.074 0.926]
    z_grid::Array{Float64,1} = [1.25, 0.2]
    nz::Int64 = length(z_grid)
end

#structure that holds model results
@everywhere mutable struct Results
    val_func::SharedArray{Float64, 2} #value function
    pol_func::SharedArray{Float64, 2} #policy function
end

#function for initializing model primitives and results
@everywhere function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = SharedArray{Float64}(zeros(prim.nk, prim.nz)) #initial value function guess
    pol_func = SharedArray{Float64}(zeros(prim.nk, prim.nz)) #initial policy function guess
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

#Bellman Operator
@everywhere function Bellman(prim::Primitives,res::Results)
    @unpack β, δ, α, k_grid, nk, markov, z_grid, nz = prim #unpack model primitives
    v_next = SharedArray{Float64}(zeros(nk, nz)) #next guess of value function to fill

    #@sync @distributed for i = 1:nz*nk
    @sync @distributed for i = 1:nz*nk
        z_index = mod(i, 2) + 1
        k_index = mod(ceil(Int64, i / nz), nk) + 1
        z = z_grid[z_index]
        k = k_grid[k_index] #value of k
            
        candidate_max = -Inf #bad candidate max
        budget = z * k^α + (1-δ)*k #budget

        for kp_index in 1:nk #loop over possible selections of k', exploiting monotonicity of policy function
            c = budget - k_grid[kp_index] #consumption given k' selection
            if c>0 #check for positivity
                val = log(c) + β * (res.val_func[kp_index, 1] * markov[z_index, 1] + res.val_func[kp_index, 2] * markov[z_index, 2]) # (res.val_func[kp_index,:] ⋅ markov[z_index,:]) #compute value
                if val>candidate_max #check for new max value
                    candidate_max = val #update max value
                    res.pol_func[k_index, z_index] = k_grid[kp_index] #update policy function
                end
            end
        end
        v_next[k_index, z_index] = candidate_max #update value function
    end
    v_next #return next guess of value function
end

#Value function iteration
@everywhere function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter

    while err>tol #begin iteration
        v_next = Bellman(prim, res) #spit out new vectors
        e1 = abs.(maximum(v_next[:,1].-res.val_func[:,1])) / abs(v_next[prim.nk, 1])
        e2 = abs.(maximum(v_next[:,2].-res.val_func[:,2])) / abs(v_next[prim.nk, 2])
        err = max(e1, e2) #reset error level
        res.val_func = v_next #update value function
        n+=1
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
@everywhere function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end
##############################################################################
