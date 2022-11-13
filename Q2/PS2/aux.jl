# PS2: Numerical Integration
# Some auxiliar calculations

# Generates Halton sequence of length n using base
function halton(base::Int64, n::Int64)
    m, d = 0, 1
    burn_in = 200
    result = zeros(burn_in + n)

    for i = 1:(burn_in + n)
        x = d - m
        if x == 1
            m = 1
            d *= base
        else
            y = d / base
            while x <= y
                y /= base
            end
            m = (base + 1) * y - x
        end
        result[i] = m / d
    end
    result[(burn_in + 1):(burn_in + n)]
end

# Returns three iid uniform(0, 1) RVs using halton for GHK
function initialize_ghk(;use_halton = true)
    n_trials = 100

    # pulls independent uniform shocks
    if use_halton

        u₀ = halton(5,  n_trials)
        u₁ = halton(7,  n_trials)
        u₂ = halton(11, n_trials)

    else

        uniform_distibution = Uniform(0, 1)

        Random.seed!(1)
        u₀ = rand(uniform_distibution, n_trials)

        Random.seed!(2)
        u₁ = rand(uniform_distibution, n_trials)

        Random.seed!(3)
        u₂ = rand(uniform_distibution, n_trials)

    end

    return [u₀, u₁, u₂]
end

# Returns correlated ε for accept-reject
function initialize_accept_reject(ρ::Float64; use_halton = true)

    u₀, u₁, u₂ = initialize_ghk(;use_halton)

    n_trials = length(u₀)

    # initialize vectors to store normal shocks
    η₀ = zeros(n_trials)
    η₁ = zeros(n_trials)
    η₂ = zeros(n_trials)

    # uses Φ_inverse function to transform from uniform to normal
    for i in 1:n_trials
        η₀[i] = Φ_inverse(u₀[i])
        η₁[i] = Φ_inverse(u₁[i])
        η₂[i] = Φ_inverse(u₂[i])
    end

    # Define correlated errors
    ε₀ = η₀ .* (1/(1 - ρ))
    ε₁ = ρ .* ε₀ .+ η₁
    ε₂ = ρ .* ε₁ .+ η₂

    return [ε₀, ε₁, ε₂]
end

# reads in grid points for quadrature integration
function initialize_quadrature_integration()

    # quadrature nodes and weights
    KPU_1d = DataFrame(CSV.File("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS2/KPU_d1_l20.csv", header=0))
    KPU_2d = DataFrame(CSV.File("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS2/KPU_d2_l20.csv", header=0))

    return [KPU_1d, KPU_2d]
end

# compute likelihood for the matrix
function likelihood(α₀::Float64, α₁::Float64, α₂::Float64,  β::Array{Float64, 1}, γ::Float64, ρ::Float64,
    t::Array{Float64, 2}, x::Array{Float64, 2}, z::Array{Float64, 2},  KPU_1d, KPU_2d,
    u₀::Array{Float64, 1}, u₁::Array{Float64, 1}, u₂::Array{Float64, 1},
    ε₀::Array{Float64, 1}, ε₁::Array{Float64, 1}, ε₂::Array{Float64, 1}; method = "quadrature")

    N = size(x)[1]

    # uses distributed for loop; on a test with quadrature, this took about 2 minutes for the dataset
    result = zeros(N)

    if method == "quadrature"
        println("Evaluating likelihoods using quadrature integration method...")

        for i = 1:N
            result[i] = likelihood_quadrature(α₀, α₁, α₂, β, γ, ρ, t[i], x[i,:], z[i,:], KPU_1d, KPU_2d)
        end

    elseif method == "ghk"
        println("Evaluating likelihoods using GHK method...")

        for i = 1:N
            result[i] = likelihood_ghk(α₀, α₁, α₂, β, γ, ρ, t[i], x[i,:], z[i,:], u₀, u₁, u₂)
        end

    elseif method == "accept_reject"
        println("Evaluating likelihoods using accept-reject method...")

        for i = 1:N
            result[i] = likelihood_accept_reject(α₀, α₁, α₂, β, γ, ρ, t[i], x[i,:], z[i,:], ε₀, ε₁, ε₂)
        end
    else
        error("Specify valid method.")
    end

    return result
end

function log_likelihood(θ::Array{Float64, 1}, t::Array{Float64, 2}, x::Array{Float64, 2}, z::Array{Float64, 2},  KPU_1d, KPU_2d,
    u₀::Array{Float64, 1}, u₁::Array{Float64, 1}, u₂::Array{Float64, 1},
    ε₀::Array{Float64, 1}, ε₁::Array{Float64, 1}, ε₂::Array{Float64, 1}; method = "quadrature")

    K_x = size(x)[2]

    α₀ = θ[1]
    α₁ = θ[2]
    α₂ = θ[3]
    β   = θ[4:(K_x+3)]
    γ   = θ[K_x+4]
    ρ   = θ[K_x+5]

    ll = sum(log.(likelihood(α₀, α₁, α₂, β, γ, ρ, t, x, z, KPU_1d, KPU_2d, u₀, u₁, u₂, ε₀, ε₁, ε₂; method = method)))

    println("Log-likelihood = ", ll)

    println("α_0 = ", α₀)
    println("α_1 = ", α₁)
    println("α_2 = ", α₂)
    println("β = ", β)
    println("γ = ", γ)
    println("ρ = ", ρ)

    println("******************************************************")

    return ll
end
