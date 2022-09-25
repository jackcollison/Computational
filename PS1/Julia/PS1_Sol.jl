########Value Function Iteration
########Econ 899: Computational Methods
########Last modified by Stefano Lord-Medrano

using Parameters, Plots, LaTeXStrings, Distributed #read in necessary packages

#Model primitives
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate.
    α::Float64 = 0.36 #capital share
    δ::Float64 = 0.025 #capital depreciation
    k_grid::Array{Float64,1} = collect(range(1.0, length = 1000, stop = 45.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital elements
    markov::Array{Float64,2} = [0.977 0.023; 0.074 0.926] #markov transition process
    z_grid::Array{Float64,1} = [1.25, 0.2] #productivity state grid
    nz::Int64 = length(z_grid) #number of productivity states
end

#Mutable struct to hold model results (vectors that will contain solutions to value function iteration)
mutable struct Results
    val_func::Array{Float64,2}
    pol_func::Array{Float64,2}
end

#Function that executes the model and returns results
 function Solve_model()
    prim = Primitives() #initialize primitives
    val_func = zeros(prim.nk, prim.nz) #preallocate value function as a vector of zeros
    pol_func = zeros(prim.nk, prim.nz) #preallocate value function as a vector of zeros
    res = Results(val_func, pol_func) #initialize results
    V_iterate(prim, res) #value function iteration
    prim, res #return deliverables
end

#Value function iteration program.
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-3)
    error = 100 #starting error
    n = 0 #counter
    while error>tol #main convergence loop
        n+=1
        v_next = Bellman(prim, res) #execute Bellman operator
        error = maximum(abs.(v_next - res.val_func)) #reset error term
        res.val_func = v_next #update value function held in results vector
    end
    println("Value functions converged in ", n, " iterations.")
end

#Bellman operator
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, α, nz, nk, z_grid, k_grid, markov = prim #unpack parameters from prim struct. Improves readability.
    v_next = zeros(nk, nz)

     for i_k = 1:nk, i_z = 1:nz #loop over state space
        candidate_max = -1e10 #something crappy
        k, z = k_grid[i_k], z_grid[i_z] #convert state indices to state values
        budget = z*k^α + (1-δ)*k #budget given current state. Doesn't this look nice?

        for i_kp = 1:nk #loop over choice of k_prime
            kp = k_grid[i_kp]
            c = budget - kp #consumption
            if c>0 #check to make sure that consumption is positive
                val = log(c) + β * sum(res.val_func[i_kp,:].*markov[i_z, :])
                if val>candidate_max #check for new max value
                    candidate_max = val
                    res.pol_func[i_k, i_z] = kp #update policy function
                end
            end
        end
        v_next[i_k, i_z] = candidate_max #update next guess of value function
    end
    v_next
end

@time prim, res = Solve_model() #solve for policy and value functions


#unpack our results and make some plots
@unpack val_func, pol_func = res
@unpack k_grid = prim

#Change in policy functions
dif_val_func_gs = diff(val_func[:, 1])
dif_val_func_bs = diff(val_func[:, 2])
dif_pol_func_gs = diff(pol_func[:, 1])
dif_pol_func_bs = diff(pol_func[:, 2])

dif_val_func=[reshape(dif_val_func_gs, 1, :); reshape(dif_val_func_bs, 1, :)]
dif_val_func=[reshape(dif_pol_func_gs, 1, :); reshape(dif_pol_func_bs, 1, :)]

#Plot value function
plot(k_grid,pol_func, label = [L"Good\; State" L"Bad\; State"], legend=:bottomright,ylabel = L"k'(k)",xlabel = L"k")
Plots.savefig("/Users/smlm/Downloads/Policy_Functions.png")
plot(k_grid,val_func, label = [L"Good\; State" L"Bad\; State"], legend=:bottomright,ylabel = L"V(k)",xlabel = L"k")
Plots.savefig("/Users/smlm/Downloads/Value_Functions.png")
#Plot changes in value function and policy function
plot(dif_pol_func_gs)
