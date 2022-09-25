##################################################
################## DIAGNOSTICS ###################
##################################################

# Import required packages
using Printf

# Include model
include("Model.jl")

# Compute welfare
function ComputeWelfare(res::Results)
    # Welfare analysis
    W = res.F .* res.value_func
    sum(W[isfinite.(W)])
end

# Compute coefficient of variation
function ComputeCV(res::Results)
    # Unpack primitives
    @unpack N, n, Jᴿ, σ, β, η, Π, Π₀, α, δ, A, na = Primitives()

    # Reshape wealth for matrix multiplication
    W = reshape(repeat(A, res.nz * N), N, na, res.nz)

    # Return coefficient of variation
    sqrt(sum(res.F .* W .^2) - sum(res.F .* W)^2) / sum(res.F .* W)
end

# Output formatting
function FormatResults(res::Results)
    # Formatting
    println("*****************************************************")
    @printf "Social Security (θ)             = %.2f\n" res.θ
    @printf "Uncertainty (Z)                 = %s\n" ifelse(size(res.Z, 1) > 1, "Yes", "No")
    @printf "Labor Disutility (γ)            = %-8.2f\n" res.γ
    @printf "Capital (K)                     = %-8.6f\n" res.K
    @printf "Labor (L)                       = %-8.6f\n" res.L
    @printf "Wage (w)                        = %-8.6f\n" res.w
    @printf "Interest (r)                    = %-8.6f\n" res.r
    @printf "Penstion Benefit (b)            = %-8.6f\n" res.b
    @printf "Total Welfare (W)               = %-8.6f\n" ComputeWelfare(res)
    @printf "Coefficient of Variation (CV)   = %-8.6f\n" ComputeCV(res)
    println("*****************************************************")
end