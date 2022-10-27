# PS7: Simulated Method of Moments
# Fall 2022

# Packages
using Parameters, Distributions, Random, LinearAlgebra, Optim, Interpolations, LaTeXStrings

## Simulation of "true data"
# Draw shocks
function shock(T::Int64, H::Int64; seed = missing)
    if !ismissing(seed)
        Random.seed!(seed)
    end
    rand(Normal(0,1),T,H)
end

# AR(1) process
function ar(ε::Array{Float64,2},ρ::Float64,σ::Float64)
    T = size(ε)[1]  # Observations
    H = size(ε)[2]  # Number of simulations

    # Initialize vector
    x = zeros(T, H)
    # Fill vector of AR(1) process given parameters ρ₀ and σ₀
    for i_H = 1:H
        x[1,i_H] = ε[1,i_H]
        for i = 2:T
            x[i,i_H]=ρ*x[i-1,i_H]+σ*ε[i,i_H]
        end
    end
    x
end

# Compute asymptotic moments
function asy_m(x::Array{Float64,2})
    T = size(x)[1]
    H = size(x)[2]
    # Mean
    x̄ = repeat(sum(x;dims=1)./T,T)
    # Lag
    x₋ = vcat(zeros(H)',x[1:T-1,:])
    # Vector of moments m₃(b)
    asy_m = zeros(T,H,3)
    asy_m[:,:,1] = x
    asy_m[:,:,2] = (x.-x̄).^2
    asy_m[:,:,3] = (x.-x̄).*(x₋.-x̄)

    asy_m
end

# Compute sample moments
function m(asy_m::Array{Float64,3})
    T = size(asy_m)[1]
    H = size(asy_m)[2]
    m = reshape(sum(asy_m; dims=1:2)/(T*H),3)
end

# Structure for parameters
@with_kw struct Primitives
    T::Int64    = 200   # Number of periods
    H::Int64    = 10    # Number of simulation
    Lʲ::Int64   = 4     # Number of lags
    nd::Float64   = 1e-10 # Numerical derivative
    ρ_g::Array{Float64,1} = 0.35:0.01:0.65
    σ_g::Array{Float64,1} = 0.80:0.01:1.20
end

# Structure for initial conditions
mutable struct DGP
    ρ₀::Float64
    σ₀::Float64
    ε::Array{Float64,2}
    x₀::Array{Float64,2}
    asy_m₀::Array{Float64,3}
    M_T::Array{Float64,1}
end

# Structure for estimation
mutable struct Estimation
    moments::Array{Int64,1}       # moments for objective function
    e::Array{Float64,2}             # shocks simulated
    ρ̂₁::Float64                     # first-stage estimator for ρ
    σ̂₁::Float64                     # first-stage estimator for σ
    ρ̂ₛ¹::Float64                     # first-stage s.e. for ρ
    σ̂ₛ¹::Float64                     # first-stage s.e. for σ
    𝐽₁::Array{Float64,2}             # first-stage Jacobian matrix
    𝕎::Array{Float64,2}             # estimator for optimal weighting matrix
    ρ̂₂::Float64                     # second-stage estimator for ρ
    σ̂₂::Float64                     # second-stage estimator for σ
    ρ̂ₛ²::Float64                    # second-stage s.e. for ρ
    σ̂ₛ²::Float64                    # second-stage s.e. for σ
    𝐽₂::Array{Float64,2}            # first-stage Jacobian matrix
    𝐉::Float64                      # Sargan J-test statistic
    p_value_𝐉::Float64              # p-value for Sargan test
end

# Simulation
function Initialize_True_Data(; seed = missing)
    ρ₀ = 0.5
    σ₀ = 1.0
    ε = shock(200,1; seed = seed)
    x₀ = ar(ε,ρ₀,σ₀)
    m₀ = asy_m(x₀)
    M_T = m(m₀)

    DGP(ρ₀,σ₀,ε,x₀,m₀,M_T)
end

function Initialize_Estimation(moments; seed = missing)
    @unpack T, H = Primitives()
    e = shock(T,H; seed = seed)
    ρ̂₁ = 0
    σ̂₁ = 0
    ρ̂ₛ¹ = 0
    σ̂ₛ¹ = 0
    𝐽₁ = zeros(2,2)
    𝕎 = zeros(length(moments),length(moments))
    ρ̂₂ = 0
    σ̂₂ = 0
    ρ̂ₛ² = 0
    σ̂ₛ² = 0
    𝐽₂ = zeros(2,2)
    𝐉 = 0
    p_value_𝐉 = 0

    Estimation(moments, e, ρ̂₁, σ̂₁, ρ̂ₛ¹, σ̂ₛ¹, 𝐽₁, 𝕎, ρ̂₂, σ̂₂, ρ̂ₛ², σ̂ₛ², 𝐽₂, 𝐉, p_value_𝐉)
end

# Compute objective function
function J_TH(b, e, moments, W, M_T)
    ρ = b[1]
    σ = b[2]

    M_TH = m(asy_m(ar(e, ρ, σ)))
    (M_T[moments]-M_TH[moments])' * W * (M_T[moments]-M_TH[moments])
end

# 3D plots
function plot_3d_J_TH(R::Estimation, D::DGP, stage::Int64)
    @unpack ρ_g, σ_g = Primitives()

    if stage == 1
        W = I
        title = string("First-Stage Objective Function: ", join(string.("asy_m",R.moments," ")))
    elseif stage == 2
        W = R.𝕎
        title = string("Second-Stage Objective Function: ", join(string.("asy_m",R.moments," ")))
    else
        error("Specify valid stage arg.")
    end

    function J_TH_plot(ρ,σ)
        return J_TH([ρ,σ], R.e, R.moments, W, D.M_T)
    end

    surface(ρ_g,σ_g, J_TH_plot,opacity=0.7,c = :blues);
    xlabel!(L"ρ");
    ylabel!(L"σ");
end

# Coefficient estimation
function b̂(R::Estimation, D::DGP, stage::Int64)
    if stage == 1
        W = I
    elseif stage == 2
        W = R.𝕎
    else
        error("Specify valid stage arg")
    end

    opt = optimize(b->J_TH(b, R.e, R.moments, W, D.M_T), [D.ρ₀,D.σ₀])
    Optim.minimizer(opt)
end

# Estimation of optimal weighting matrix 𝕎
function 𝕎(R::Estimation)
    @unpack T, H, Lʲ = Primitives()

    asy_m_1 = asy_m(ar(R.e, R.ρ̂₁, R.σ̂₁))
    M_TH_1 = m(asy_m_1)

    function compute_Γ(j)
        Γ = zeros(3,3)
        for i_1 = 1:3
            for i_2 in 1:3
                Γ[i_1,i_2]=sum((asy_m_1[j+1:T,:,i_1].-M_TH_1[i_1])*(asy_m_1[1:T-j,:,i_2].-M_TH_1[i_2])')/(T*H)
            end
        end
        Γ[R.moments, R.moments]
    end
    S_TH = compute_Γ(0)
    for j = 1:Lʲ
        Γ_j = compute_Γ(j)
        S_TH += (1-(j/(Lʲ+1)))*(Γ_j+Γ_j')
    end
    S_TH = (1+1/H)*S_TH
    inv(S_TH)
end

# Computation of numerical derivative
function jacobian(R::Estimation, stage::Int64)
    @unpack nd = Primitives()

    # choose weighting matrix based on stage argument
    if stage == 1
        ρ = R.ρ̂₁
        σ = R.σ̂₁

    elseif stage == 2
        ρ = R.ρ̂₂
        σ = R.σ̂₂
    else
        error("Specify valid stage arg.")
    end

    M_TH   = m(asy_m(ar(R.e, ρ, σ)))
    M_TH_ρ = m(asy_m(ar(R.e, ρ - nd, σ)))
    M_TH_σ = m(asy_m(ar(R.e, ρ, σ - nd)))

    jacobian_ρ = - (M_TH - M_TH_ρ) / nd
    jacobian_σ = - (M_TH - M_TH_σ) / nd

    return(hcat(jacobian_ρ[R.moments], jacobian_σ[R.moments]))
end

# Computation of S.E.
function se(R::Estimation, stage::Int64)
    @unpack T = Primitives()
    if stage == 1
        W = I
        jacobian = R.𝐽₁
    elseif stage == 2
        W = R.𝕎
        jacobian = R.𝐽₂
    else
        error("Specify valid stage arg.")
    end
    sqrt.(diag((1/T) * inv(jacobian' * W * jacobian)))
end

# Computation of Sargan 𝐉 statistic
function j_test(R::Estimation, D::DGP)
    @unpack T, H = Primitives()

    stat = T * ( H / (1 + H)) * J_TH([R.ρ̂₂, R.σ̂₂], R.e, R.moments, R.𝕎, D.M_T)
    p_value = cdf(Chisq(1), stat)

    return [stat, p_value]
end

# Estimation
function estimate(moments; D_seed = missing, R_seed = missing)
    # Initialize results objects
    D = Initialize_True_Data(;seed = D_seed)
    R = Initialize_Estimation(moments; seed = R_seed)

    # First-stage estimation
    b̂₁   = b̂(R, D, 1)
    R.ρ̂₁ = b̂₁[1]
    R.σ̂₁ = b̂₁[2]

    # S.E. computation
    R.𝐽₁ = jacobian(R, 1)
    R.ρ̂ₛ¹   = se(R, 1)[1]
    R.σ̂ₛ¹   = se(R, 1)[2]

    # Estimate optimal weighting matrix 𝕎
    R.𝕎 = 𝕎(R)

    # Second-stage estimation
    b̂₂   = b̂(R, D, 2)
    R.ρ̂₂ = b̂₂[1]
    R.σ̂₂ = b̂₂[2]

    # S.E. computation
    R.𝐽₂ = jacobian(R, 2)
    R.ρ̂ₛ²   = se(R, 2)[1]
    R.σ̂ₛ²   = se(R, 2)[2]

    # 𝐉 test
    R.𝐉   = j_test(R, D)[1]
    R.p_value_𝐉 = j_test(R, D)[2]

    R
end

# Bootstrap
function bootstrap_se(moments)
    n_bs = 1000

    bs_results = zeros(n_bs, 4)
    for i = 1:n_bs
        println(i)

        # Initialize results objects
        D = Initialize_True_Data()
        R = Initialize_Estimation(moments)

        # First stage
        b̂₁   = b̂(R, D, 1)
        R.ρ̂₁ = b̂₁[1]
        R.σ̂₁ = b̂₁[2]

        # Estimate optimal weighting matrix
        R.𝕎 = 𝕎(R)

        # second stage
        b̂₂   = b̂(R, D, 2)

        bs_results[i, 1] = b̂₁[1]
        bs_results[i, 2] = b̂₂[1]
        bs_results[i, 3] = b̂₁[2]
        bs_results[i, 4] = b̂₂[2]
    end
    bs_results
end

# Report estimates in Table
function process_results(R::Estimation)
    [R.ρ̂₁, R.ρ̂ₛ¹, R.σ̂₁, R.σ̂ₛ¹, R.ρ̂₂, R.ρ̂ₛ², R.σ̂₂, R.σ̂ₛ², R.𝐉, R.p_value_𝐉]
end

function create_table(results_vector::Array{Estimation})

    # create matrix of results
    temp = reduce(hcat,process_results.(results_vector))'

    # convert into data frame
    table = DataFrame(Tables.table(temp))
    rename!(table, [:rho_hat_1, :rho_se_1, :sigma_hat_1, :sigma_se_1, :rho_hat_2, :rho_se_2, :sigma_hat_2, :sigma_se_2, :j_test_stat, :j_test_p_value])
end