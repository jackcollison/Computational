using Parameters, Interpolations, Statistics, Random, Distributions, Optim

## Orginally prepared by Philip Coyle
## Modified by Michael Nattinger
## Notation changes in transition matrices by Yeonggyu Yun

## Housekeeping
@with_kw struct Params
    cBET::Float64 = 0.99
    cALPHA::Float64 = 0.36
    cDEL::Float64 = 0.025
    cLAM::Float64 = 0.5

    N::Int64 = 5000
    T::Int64 = 11000
    burn::Int64 = 1000

    tol_vfi::Float64 = 1e-4
    tol_coef::Float64 = 1e-4
    tol_r2::Float64 = 1.0 - 1e-2
    maxit::Int64 = 10000
end

@with_kw struct Grids
    k_lb::Float64 = 0.001
    k_ub::Float64 = 20.0
    n_k::Int64 = 21
    k_grid::Array{Float64,1} = range(k_lb, stop = k_ub, length = n_k)

    K_lb::Float64 = 10.0
    K_ub::Float64 = 15.0
    n_K::Int64 = 11
    K_grid::Array{Float64,1} = range(K_lb, stop = K_ub, length = n_K)

    eps_l::Float64 = 0.0
    eps_h::Float64 = 0.3271
    n_eps::Int64 = 2
    eps_grid::Array{Float64,1} = [eps_h, eps_l]

    z_l::Float64 = 0.99
    z_h::Float64 = 1.01
    n_z::Int64 = 2
    z_grid::Array{Float64,1} = [z_h, z_l]
end

@with_kw struct Shocks
    #parameters of transition matrix:
    d_ug::Float64 = 1.5 # Unemp Duration (Good Times)
    u_g::Float64 = 0.04 # Fraction Unemp (Good Times)
    d_g::Float64 = 8.0 # Duration (Good Times)
    u_b::Float64 = 0.1 # Fraction Unemp (Bad Times)
    d_b::Float64 = 8.0 # Duration (Bad Times)
    d_ub::Float64 = 2.5 # Unemp Duration (Bad Times)

    #transition probabilities for aggregate states
    pgg::Float64 = (d_g-1.0)/d_g
    pbg::Float64 = 1.0 - (d_b-1.0)/d_b
    pgb::Float64 = 1.0 - (d_g-1.0)/d_g
    pbb::Float64 = (d_b-1.0)/d_b

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (d_ug-1.0)/d_ug
    pbb00::Float64 = (d_ub-1.0)/d_ub
    pgb00::Float64 = 1.25*pbb00
    pbg00::Float64 = 0.75*pgg00

    #transition probabilities for aggregate states and becoming employed
    pgg01::Float64 = 1.0 - (d_ug-1.0)/d_ug
    pbb01::Float64 = 1.0 - (d_ub-1.0)/d_ub
    pgb01::Float64 = 1.0 - 1.25*pbb00
    pbg01::Float64 = 1.0 - 0.75*pgg00

    #transition probabilities for aggregate states and becoming unemployed
    pgg10::Float64 = (u_g - u_g*pgg00)/(1.0-u_g)
    pbb10::Float64 = (u_b - u_b*pbb00)/(1.0-u_b)
    pbg10::Float64 = (u_g - u_b*pbg00)/(1.0-u_b)
    pgb10::Float64 = (u_b - u_g*pgb00)/(1.0-u_g)

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - (u_g - u_g*pgg00)/(1.0-u_g)
    pbb11::Float64 = 1.0 - (u_b - u_b*pbb00)/(1.0-u_b)
    pgb11::Float64 = 1.0 - (u_b - u_g*pgb00)/(1.0-u_g)
    pbg11::Float64 = 1.0 - (u_g - u_b*pbg00)/(1.0-u_b)

    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg10
                            pgg01 pgg00]

    Mgb::Array{Float64,2} = [pgb11 pgb10
                            pgb01 pgb00]

    Mbg::Array{Float64,2} = [pbg11 pbg10
                            pbg01 pbg00]

    Mbb ::Array{Float64,2} = [pbb11 pbb10
                             pbb01 pbb00]

    markov::Array{Float64,2} = [pgg*Mgg pgb*Mgb
                                pbg*Mbg pbb*Mbb]
end

mutable struct Results
    pf_k::Array{Float64,4}
    pf_v::Array{Float64,4}

    a0::Float64
    a1::Float64
    b0::Float64
    b1::Float64

    R2::Array{Float64,1}
end

function Initialize()
    pars = Params()
    grids = Grids()
    shocks = Shocks()


    pf_k::Array{Float64,4} = zeros(grids.n_k, grids.n_eps, grids.n_K, grids.n_z)
    pf_v::Array{Float64,4} = zeros(grids.n_k, grids.n_eps, grids.n_K, grids.n_z)
    a0::Float64 = 0.095
    a1::Float64 = 0.999
    b0::Float64 = 0.085
    b1::Float64 = 0.999
    R2::Array{Float64,1} = zeros(2)
    res = Results(pf_k, pf_v, a0, a1, b0, b1, R2)
    pars, grids, shocks, res
end

function draw_shocks(S::Shocks, N::Int64,T::Int64)
    @unpack pgg, pbb, Mgg, Mgb, Mbg, Mbb = S

    # Shock
    Random.seed!(12032020)
    dist = Uniform(0, 1)

    # Allocate space for shocks and initialize
    idio_state = zeros(N,T)
    agg_state = zeros(T)
    idio_state[:,1] .= 1
    agg_state[1] = 1

    for t = 2:T
        agg_shock = rand(dist)
        if agg_state[t-1] == 1 && agg_shock < pgg
            agg_state[t] = 1
        elseif agg_state[t-1] == 1 && agg_shock > pgg
            agg_state[t] = 2
        elseif agg_state[t-1] == 2 && agg_shock < pbb
            agg_state[t] = 2
        elseif agg_state[t-1] == 2 && agg_shock > pbb
            agg_state[t] = 1
        end

        for n = 1:N
            idio_shock = rand(dist)
            if agg_state[t-1] == 1 && agg_state[t] == 1
                p11 = Mgg[1,1]
                p00 = Mgg[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            elseif agg_state[t-1] == 1 && agg_state[t] == 2
                p11 = Mgb[1,1]
                p00 = Mgb[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            elseif agg_state[t-1] == 2 && agg_state[t] == 1
                p11 = Mbg[1,1]
                p00 = Mbg[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            elseif agg_state[t-1] == 2 && agg_state[t] == 2
                p11 = Mbb[1,1]
                p00 = Mbb[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            end
        end
    end

    return idio_state, agg_state
end

function Bellman(P::Params, G::Grids, S::Shocks, R::Results)
    @unpack cBET, cALPHA, cDEL = P
    @unpack n_k, k_grid, n_eps, eps_grid, eps_h, K_grid, n_K, n_z, z_grid = G
    @unpack u_g, u_b, markov = S
    @unpack pf_k, pf_v, a0, a1, b0, b1= R

    pf_k_up = zeros(n_k, n_eps, n_K, n_z)
    pf_v_up = zeros(n_k, n_eps, n_K, n_z)

    # In Julia, this is how we define an interpolated function.
    # Need to use the package "Interpolations".
    # (If you so desire, you can write your own interpolation function too!)
    k_interp = interpolate(k_grid, BSpline(Linear()))
    v_interp = interpolate(pf_v, BSpline(Linear()))

    for (i_z, z_today) in enumerate(z_grid)
        for (i_K, K_today) in enumerate(K_grid)
            if i_z == 1
                K_tomorrow = a0 + a1*log(K_today)
                L_today = (1 - u_g) * eps_h
            elseif i_z == 2
                K_tomorrow = b0 + b1*log(K_today)
                L_today = (1 - u_b) * eps_h
            end
            K_tomorrow = exp(K_tomorrow)

            # See that K_tomorrow likely does not fall on our K_grid...this is why we need to interpolate!
            i_Kp = get_index(K_tomorrow, K_grid)

            r_today = z_today * cALPHA * K_today^(cALPHA - 1) * L_today^(1 - cALPHA)
            w_today = z_today * (1 - cALPHA) * K_today^cALPHA * L_today^(-cALPHA)

            for (i_eps, eps_today) in enumerate(eps_grid)
                row = i_eps + n_eps*(i_z-1)

                for (i_k, k_today) in enumerate(k_grid)
                    budget_today = r_today*k_today + w_today*eps_today + (1.0 - cDEL)*k_today

                    # We are defining the continuation value. Notice that we are interpolating over k and K.
                    v_tomorrow(i_kp) = markov[row,1]*v_interp(i_kp,1,i_Kp,1) + markov[row,2]*v_interp(i_kp,2,i_Kp,1) +
                                        markov[row,3]*v_interp(i_kp,1,i_Kp,2) + markov[row,4]*v_interp(i_kp,2,i_Kp,2)

                    # We are now going to solve the HH's problem (solve for k).
                    # We are defining a function val_func as a function of the agent's capital choice.
                    val_func(i_kp) = log(budget_today - k_interp(i_kp)) +  cBET*v_tomorrow(i_kp)

                    # Need to make our "maximization" problem a "minimization" problem.
                    obj(i_kp) = -val_func(i_kp)
                    lower = 1.0
                    upper = get_index(budget_today, k_grid)

                    # Then, we are going to maximize the value function using an optimization routine.
                    # Note: Need to call in optimize to use this package.
                    opt = optimize(obj, lower, upper)

                    k_tomorrow = k_interp(opt.minimizer[1])
                    v_today = -opt.minimum

                    # Update PFs
                    pf_k_up[i_k, i_eps, i_K, i_z] = k_tomorrow
                    pf_v_up[i_k, i_eps, i_K, i_z] = v_today
                end
            end
        end
    end

    return pf_k_up, pf_v_up
end

function get_index(val::Float64, grid::Array{Float64,1})
    n = length(grid)
    index = 0
    if val <= grid[1]
        index = 1
    elseif val >= grid[n]
        index = n
    else
        index_upper = findfirst(x->x>val, grid)
        index_lower = index_upper - 1
        val_upper, val_lower = grid[index_upper], grid[index_lower]

        index = index_lower + (val - val_lower) / (val_upper - val_lower)
    end
    return index
end

function VFI(P::Params, G::Grids, S::Shocks, R::Results, err::Float64 = 1000.0)
    @unpack tol_vfi = P

    while err > tol_vfi
        pf_k_up, pf_v_up = Bellman(P, G, S, R)
        err = maximum(abs.(pf_v_up .- R.pf_v))

        R.pf_k = pf_k_up
        R.pf_v = pf_v_up
    end
end

function run_KS(P::Params, G::Grids, S::Shocks, R::Results, err_coef::Float64 = 100.0)
    @unpack N, T, burn, tol_coef, tol_r2, maxit = P
    counter = 0
    Ks = zeros(T)

    idio_state, agg_state = draw_shocks(S, N, T)

    while counter < maxit
        while minimum(R.R2) < tol_r2 || err_coef > tol_coef

            VFI(P, G, S, R) # value function iteration

            ks_today = ones(N) .* 10.55 # initial values of small k's
            ks_tomorrow = zeros(N) # store new small k's

            k_interp = interpolate(R.pf_k, BSpline(Linear()))

            for t in 1:T # simulate path of aggregate capital
                if t == 1
                    for n in 1:N
                        i_K = get_index(10.55, G.K_grid)
                        i_k = get_index(ks_today[n], G.k_grid)

                        ks_tomorrow[n] = k_interp(i_k, idio_state[n,t], i_K, agg_state[t])
                    end
                    Ks[t] = mean(ks_tomorrow)
                    ks_today = ks_tomorrow
                elseif t > 1
                    for n in 1:N
                        i_K = get_index(Ks[t-1], G.K_grid)
                        i_k = get_index(ks_today[n], G.k_grid)

                        ks_tomorrow[n] = k_interp(i_k, idio_state[n,t], i_K, agg_state[t])
                    end
                    Ks[t] = mean(ks_tomorrow)
                    ks_today = ks_tomorrow
                end
            end

            # regression... Julia grammar sucks... will change once I get a better idea how to separate K's into two matrices
            agg_state_reg = agg_state[(burn+1):(T-1)]
            Ks_now = Ks[(burn+1):(T-1)]
            Ks_next = Ks[(burn+2):T]

            Yg = log.(Ks_next[findall(i -> (i == 1), agg_state_reg)])
            Xg = log.(Ks_now[findall(i -> (i == 1), agg_state_reg)])
            Yb = log.(Ks_next[findall(i -> (i == 2), agg_state_reg)])
            Xb = log.(Ks_now[findall(i -> (i == 2), agg_state_reg)])

            a1_new = (sum(Xg.*Yg) - mean(Xg) * sum(Yg)) / (sum(Xg.*Xg)- sum(Xg) * mean(Xg))
            a0_new = mean(Yg) - a1_new * mean(Xg)

            b1_new = (sum(Xb.*Yb) - mean(Xb) * sum(Yb)) / (sum(Xb.*Xb)- sum(Xb) * mean(Xb))
            b0_new = mean(Yb) - b1_new * mean(Xb)

            R.R2[1] = 1 - sum((Yg .- a0_new .- a1_new .* Xg).^2) / sum((Yg .- mean(Yg)).^2)
            R.R2[2] = 1 - sum((Yb .- b0_new .- b1_new .* Xb).^2) / sum((Yb .- mean(Yb)).^2)

            err_coef = abs(R.a0-a0_new) + abs(R.a1-a1_new) + abs(R.b0-b0_new) + abs(R.b1-b1_new)
            
            R.a0 = a0_new
            R.a1 = a1_new
            R.b0 = b0_new
            R.b1 = b1_new

            counter += 1

            println("Iteration: ", counter, " Coefficient Error: ", err_coef, " Minimum R2: ", minimum(R.R2))
        end
        println("Last Iteration: ", counter, " Converged with Coefficient Error: ", err_coef, " and Maximum R2: ", minimum(R.R2))
    end
end
