#parameters
@with_kw struct Primitives
    β::Float64 = 0.9932 #discount rate
    α::Float64 = 1.5 #risk aversion
    Es::Array{Float64} = [1.0, 0.5] #earnings
    a_min::Float64 = -2 #asset lower bound
    a_max::Float64 = 5 #asset upper bound
    length_a_grid::Int64 = 1000 #number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a_min, length = length_a_grid, stop = a_max)) #capital grid
    Π::Matrix{Float64} = [0.97 0.5; 0.03 0.5] #transpose of original transition matrix
end

#structure that holds model results
mutable struct Results
    val_func::Matrix{Float64} #value function
    pol_func::Matrix{Float64} #policy function
    q::Float64 #equilibrium bond price
    mu::Matrix{Float64}
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = reshape(zeros(2*prim.length_a_grid), prim.length_a_grid, 2) #initial value function guess
    pol_func = reshape(zeros(2*prim.length_a_grid), prim.length_a_grid, 2) #initial policy function guess
    q = 0.9932 #initial bond price to guess
    mu = (prim.a_grid .- (prim.a_min) ./ (prim.a_max+prim.a_min)) * [0.9434 0.0566] # initial guess for mu
    res = Results(val_func, pol_func, q, mu) #initialize results struct
    prim, res #return deliverables
end

# T operator
function Bellman(prim::Primitives,res::Results)
    @unpack q, val_func = res #unpack value function
    @unpack a_grid, β, α, length_a_grid, Π, Es = prim #unpack model primitives
    v_next = reshape(zeros(2*length_a_grid), length_a_grid, 2) #next guess of value function to fill
    ctmp = zeros(length_a_grid, 2) #consumption matrix to fill
    vtmp = zeros(length_a_grid, length_a_grid, 2)

    for a_index = 1:length_a_grid, j = 1:2
        a = a_grid[a_index] #value of k
        ctmp[:,j] = Es[j] + a .- q.*a_grid
        ctmp[:,j] = ifelse.(ctmp[:,j] .> 0, 1, 0).*ctmp[:,j]

        vtmp[a_index,:,j] = ((ctmp[:,j]).^(1-α) .- 1)./(1-α) + β*val_func[:,:]*Π[:,j]
        v_next[a_index,j] = maximum(vtmp[a_index,:,j])
        res.pol_func[a_index,j] = a_grid[findmax(vtmp[a_index,:,j])[2]]
    end
    v_next #return next guess of value function
end

#Value function iteration
function get_g(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter

    while err>tol #begin iteration
        v_next = Bellman(prim, res) #spit out new vectors
        err = abs.(maximum(v_next.-res.val_func)) #reset error level
        res.val_func = v_next #update value function
        n+=1
    end
    #println("Value function converged in ", n, " iterations.")
end

function get_mu(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    @unpack a_grid, length_a_grid, Π = prim
    ED = 1 # excess demand

    while abs(ED) > tol
        mu_new = zeros(length_a_grid, 2)

        while err>tol
            get_g(prim, res)

            for i = 1:length_a_grid, j = 1:2
                a_new = a_grid[i] # a'
                mu_new[i,j] = sum(res.mu.*(res.pol_func .== a_new) * Π[j,:])
            end
            err = maximum((mu_new .- res.mu)./res.mu)
            res.mu = mu_new
        end
        ED = sum(res.pol_func .* res.mu)
        if ED > tol
            res.q += 0.01*res.q
        elseif ED < -tol
            res.q -= 0.01*res.q
        end
    end
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    get_mu(prim, res)
end
