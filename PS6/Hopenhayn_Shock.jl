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
    U::Array{Float64, 1}
    V::Array{Float64, 2}
    Σ::Array{Float64, 2}
    cf::Float64
    α::Float64
end

function Initialize(cf::Float64, α::Float64)
    pars = Params()
    μ::Array{Float64, 1} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]
    p::Float64 = 4.5
    M::Float64 = 0.03
    U::Array{Float64, 1} = ones(pars.ns) .* 0.5772 ./ α
    V::Array{Float64, 2} = zeros(pars.ns, 2)
    Σ::Array{Float64, 2} = ones(pars.ns, 2) .* 0.5
    res = Results(μ, p, M, U, V, Σ, cf, α)
    pars, res
end

function FindU(pars, res, tol::Float64 = 1e-4)
    @unpack α, p, cf = res
    @unpack S, β, θ, F, ns = pars
    err_u = 1000.0
    while err_u > tol
        U_cand = zeros(ns)
        for (i_s, s) in enumerate(S)
            Nd = (1 / (θ * p * s))^(1 / (θ-1))

            V0 = p * s * Nd^θ - Nd - p * cf + β * sum(F[i_s,:] .* res.U)
            V1 = p * s * Nd^θ - Nd - p * cf

            c = α * max(V0, V1)

            Us = 0.5772 / α + (c + log(sum([exp(α * V0 - c), exp(α * V1 - c)]))) / α
            U_cand[i_s] = Us
            res.Σ[i_s, 1] = exp(α*V0 - c) / sum([exp(α * V0 - c), exp(α * V1 - c)])
            res.Σ[i_s, 2] = exp(α*V1 - c) / sum([exp(α * V0 - c), exp(α * V1 - c)])
        end
        err_u = maximum(abs.(res.U .- U_cand))
        res.U = U_cand
    end
end

function EntrantValue(pars, res)
    @unpack ν = pars
    @unpack U = res

    EV = sum(U .* ν)
    return EV
end

function StatDist(pars, res, tol_m::Float64 = 1e-4)
    @unpack Σ, M = res
    @unpack F, ns, ν = pars

    err_m = 1000
    new_dist = zeros(ns)

    while err_m > tol_m
        for i = 1:ns
            new_dist[i] = sum(Σ[:,1] .* F[:,i] .* res.μ) + sum(Σ[:,1] .* F[:,i] .* ν)
        end
        err_m = maximum(abs.(res.μ .- new_dist))
        res.μ = new_dist
    end
end

function GetAggLabor(pars, res)
    @unpack μ, M, p, cf = res
    @unpack A, ν, θ, S, ce = pars

    N_star = (1 ./ (θ * p .* S)).^(1/(θ-1))
    Pi_star = p .* S .* N_star.^θ .- N_star .- p * cf

    LD = sum(N_star .* μ) + sum(N_star .* ν)
    Pi_agg = sum(Pi_star .* μ) + sum(Pi_star .* ν) - M * p * ce
    LS = 1/A - Pi_agg

    return LD, LS, Pi_agg
end

function SolveModel(pars, res, tol_p::Float64 = 1e-4, tol_m::Float64 = 1e-4)
    @unpack A, ce = pars

    counter = 0
    err_p = 1000
    while err_p > tol_p
        counter += 1

        p_old = res.p
        FindU(pars, res)
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
    StatDist(pars, res)
end
