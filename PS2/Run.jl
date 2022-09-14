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
    ctmp = zeros(nk, 2) #consumption matrix to fill
    vtmp = zeros(nk, nk, 2)

    for k_index = 1:nk, j = 1:2
        k = k_grid[k_index] #value of k
        ctmp[:,j] = Z[j]*k^α + (1-δ)*k .- k_grid
        ctmp[:,j] = ifelse.(ctmp[:,j] .> 0, 1, 0).*ctmp[:,j]

        vtmp[k_index,:,j] = log.(ctmp[:,j]) + β*val_func[:,:]*Π[:,j]
        v_next[k_index,j] = maximum(vtmp[k_index,:,j])
        res.pol_func[k_index,j] = k_grid[findmax(vtmp[k_index,:,j])[2]]
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
