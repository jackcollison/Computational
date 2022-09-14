#keyword-enabled structure to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate
    δ::Float64 = 0.025 #depreciation rate
    α::Float64 = 0.36 #capital share
    k_min::Float64 = 0.01 #capital lower bound
    k_max::Float64 = 75.0 #capital upper bound
    nk::Int64 = 1000 #number of capital grid points
    k_grid::Array{Float64,1} = collect(range(k_min, length = nk, stop = k_max)) #capital grid
    Π::Matrix{Float64} = [0.977 0.074; 0.023 0.926] #transpose of original transition matrix
    Z::Vector{Float64} = [1.25, 0.2]
end

#structure that holds model results
mutable struct Results
    val_func::Matrix{Float64} #value function
    pol_func::Matrix{Float64} #policy function
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = reshape(zeros(2*prim.nk), prim.nk, 2) #initial value function guess
    pol_func = reshape(zeros(2*prim.nk), prim.nk, 2) #initial policy function guess
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

#Bellman Operator
function Bellman(prim::Primitives,res::Results)
    @unpack val_func = res #unpack value function
    @unpack k_grid, β, δ, α, nk, Π, Z = prim #unpack model primitives
    v_next = reshape(zeros(2*nk), nk, 2) #next guess of value function to fill
    c = zeros(2) #consumption matrix to fill

    for k_index = 1:nk
        k = k_grid[k_index] #value of k
        candidate_max1 = -Inf #bad candidate max
        candidate_max2 = -Inf #bad candidate max

        budget1 = Z[1]*k^α + (1-δ)*k #budget
        budget2 = Z[2]*k^α + (1-δ)*k #budget

        for kp_index in 1:nk #loop over possible selections of k', exploiting monotonicity of policy function
            c[1] = budget1 - k_grid[kp_index] #consumption given k' selection
            c[2] = budget2 - k_grid[kp_index]
            if c[1]>0 #check for positivity
                val1 = log(c[1]) + β*val_func[kp_index,:]'Π[:,1] #compute value
                if val1>candidate_max1 #check for new max value
                    candidate_max1 = val1 #update max value
                    res.pol_func[k_index,1] = k_grid[kp_index] #update policy function
                end
            end
            if c[2]>0 #check for positivity
                val2 = log(c[2]) + β*val_func[kp_index,:]'Π[:,2] #compute value
                if val2>candidate_max2 #check for new max value
                    candidate_max2 = val2 #update max value
                    res.pol_func[k_index,2] = k_grid[kp_index] #update policy function
                end
            end
        end
        v_next[k_index,:] = [candidate_max1, candidate_max2] #update value function
    end
    v_next #return next guess of value function
end

#Value function iteration
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter

    while err>tol #begin iteration
        v_next = Bellman(prim, res) #spit out new vectors
        err = abs.(maximum(v_next.-res.val_func)) #reset error level
        res.val_func = v_next #update value function
        n+=1
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end
##############################################################################
