using Parameters

@with_kw struct Params
    β::Float64 = 0.8
    θ::Float64 = 0.64
    S::Array{Float64, 1} = [3.98*1e-4, 3.58, 6.82, 12.18, 18.79]
    ν::Array{Float64, 1} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]
    ns::Int64 = 5
    ce::Float64 = 5
    F::Array{Float64, 2} = [0.6598 0.2600 0.0416 0.0331 0.0055
                            0.1997 0.7201 0.0420 0.0326 0.0056
                            0.2000 0.2000 0.5555 0.0344 0.0101
                            0.2000 0.2000 0.2502 0.3397 0.0101
                            0.2000 0.2000 0.2500 0.3400 0.0100]
    A::Float64 = 1/200
end

mutable struct Results
    μ::Array{Float64, 1}
    p::Float64
    M::Float64
    W::Array{Float64, 1}
    X::Array{Float64, 1}
    cf::Float64
    NA::Float64
    NIm::Float64
    NEn::Float64
end

function Initialize(cf::Float64)
    pars = Params()
    μ::Array{Float64, 1} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]
    p::Float64 = 4.5
    M::Float64 = 1.0
    W::Array{Float64, 1} = zeros(pars.ns)
    X::Array{Float64, 1} = zeros(pars.ns)
    NA::Float64 = 0.0
    NIm::Float64 = 0.0
    NEn::Float64 = 0.0
    res = Results(μ, p, M, W, X, cf, NA, NIm, NEn)
    pars, res
end

function Bellman(pars, res)
    @unpack p, W, cf = res
    @unpack S, β, θ, F, ns = pars

    W_cand = zeros(ns)
    X_cand = zeros(ns)
    for (i_s, s) in enumerate(S)
        Nd = (1 / (θ * p * s))^(1 / (θ-1))

        W0 = p * s * Nd^θ - Nd - p * cf + β * sum(F[i_s,:] .* W)
        W1 = p * s * Nd^θ - Nd - p * cf
        W_cand[i_s] = findmax([W0, W1])[1]
        X_cand[i_s] = findmax([W0, W1])[2]
    end

    return W_cand, X_cand
end

function VFI(pars, res, tol::Float64 = 1e-4)
    err = 1000

    while err > tol
        W_cand, X_cand = Bellman(pars, res)
        err = maximum(abs.(res.W .- W_cand))
        res.W = W_cand
        res.X = X_cand
    end
end

function EntrantValue(pars, res)
    @unpack ν = pars
    @unpack W = res

    EV = sum(W .* ν)
    return EV
end

function StatDist(pars, res, tol_m::Float64 = 1e-4)
    @unpack μ, X, M = res
    @unpack F, ns, ν = pars

    err_m = 1000
    new_dist = zeros(ns)

    while err_m > tol_m
        for i = 1:ns
            new_dist[i] = sum((2 .- X) .* F[:,i] .* μ) + M * sum((2 .- X) .* F[:,i] .* ν)
        end
        err_m = maximum(abs.(res.μ - new_dist))
        res.μ = new_dist
    end
end

function GetAggLabor(pars, res)
    @unpack μ, M, p, cf = res
    @unpack A, ν, θ, S, ce = pars

    N_star = (1 ./ (θ * p .* S)).^(1/(θ-1))
    Pi_star = p .* S .* N_star.^θ .- N_star .- p * cf

    LD = sum(N_star .* μ) + M * sum(N_star .* ν)
    Pi_agg = sum(Pi_star .* μ) + M * sum(Pi_star .* ν) - M * p * ce
    LS = 1/A - Pi_agg

    res.NA = LD
    res.NIm = sum(N_star .* μ)
    res.NEn = M * sum(N_star .* ν)

    return LD, LS, Pi_agg
end

function SolveModel(pars, res, tol_p::Float64 = 1e-4, tol_m::Float64 = 1e-4)
    @unpack A, ce = pars

    counter = 0
    err_p = 1000
    while err_p > tol_p
        counter += 1

        p_old = res.p
        VFI(pars, res)
        EV = EntrantValue(pars, res)

        err_p = abs(EV - p_old * ce)

        if EV - p_old * ce > tol_p
            res.p = p_old * 0.99
        elseif EV - p_old *ce < -tol_p
            res.p = p_old * 1.01
        end
        println("Iteration: ", counter, " Error: ", err_p)

    end

    StatDist(pars, res)
    LD, LS, Pi_agg = GetAggLabor(pars, res)

    res.M = 1 / (A *(LD + Pi_agg))
    res.μ = res.M .* res.μ

    GetAggLabor(pars, res)
end
