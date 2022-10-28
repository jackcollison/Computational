using Parameters, Distributions, Statistics, ShiftedArrays, LinearAlgebra, Optim, Random

#### Define necessary functions - notations follow Dean Corbae's problem set for SMM estimation of AR(1)

function gen_true(ρ::Float64 = 0.5, σ::Float64 = 1.0, x0::Float64 = 0.0, T::Int64 = 200)
    x = zeros(T+1)
    x[1] = x0
    for i in 2:T+1
        x[i] = ρ * x[i-1] + rand(Normal(0, σ))
    end
    return x[2:T+1]
end

function get_MT(x::Array{Float64, 1})
    MT1 = mean(x)
    MT2 = mean((x .- MT1).^2)
    MT3 = mean(skipmissing((x .- MT1) .* (ShiftedArrays.lag(x) .- MT1)))
    MT = [MT1, MT2, MT3]
    return MT
end

function gen_sim(H::Int64 = 10, T::Int64 = 200, σe::Float64 = 1.0)
    e_sim = reshape(rand(Normal(0, σe), T*H), H, T)
    return e_sim
end

function get_J(W::Array{Float64, 2}, MT::Array{Float64, 1}, use::Array{Int64, 1}, H::Int64 = 10, T::Int64 = 200, ρ::Float64 = 0.5, σ::Float64 = 1.0, e_sim::Array{Float64, 2} = e_sim)
    ys = zeros(H,T+1)

    MTs = zeros(H, 3)
    for i in 1:H
        for t in 2:T+1
            ys[i,t] = ρ .* ys[i,t-1] .+ σ * e_sim[i,t-1]
        end
        MTs[i,:] = get_MT(ys[i,2:T+1])
    end
    MTH = [mean(MTs[:,1]), mean(MTs[:,2]), mean(MTs[:,3])]

    J = reshape(MT[use] .- MTH[use], 1, length(use)) * W * reshape(MT[use] .- MTH[use], length(use), 1)
    J = J[1]
    return J
end

function get_b(flag::Int64, W::Array{Float64, 2}, MT::Array{Float64, 1}, use::Array{Int64, 1}, e_sim::Array{Float64, 2}, H::Int64 = 10, T::Int64 = 200, ρ_l::Float64 = 0.35, ρ_h::Float64 = 0.65, σ_l::Float64 = 0.8, σ_h::Float64 = 1.2)
    rhos = collect(ρ_l:0.01:ρ_h)
    sigmas = collect(σ_l:0.01:σ_h)
    Js = zeros(length(rhos), length(sigmas))

    for (i, rho_i) in enumerate(rhos), (j, sigma_j) in enumerate(rhos)
        Js[i,j] = get_J(W, MT, use, H, T, rho_i, sigma_j)
    end

    M = findmax(Js)
    bhat = [rhos[M[2][1]], sigmas[M[2][2]]]
    MTs = zeros(H, 3)
    ys = zeros(H,T+1)

    for i in 1:H
        for t in 2:T+1
            ys[i,t] = bhat[1] .* ys[i,t-1] .+ bhat[2] .* e_sim[i,t-1]
        end
        MTs[i,:] = get_MT(ys[i,2:T+1])
    end
    MTHs = [mean(MTs[:,1]), mean(MTs[:,2]), mean(MTs[:,3])]
    ys = ys[:,2:T+1]

    zs = zeros(H,T,3)
    if flag == 1
        zs[:,:,1] = ys
        zs[:,:,2] = (ys .- repeat(mean(ys, dims = 2),1, T)).^2
        zs[:,:,3] = hcat(fill(0, H), (ys[:,2:T] .- repeat(mean(ys, dims = 2),1, T-1)) .* (ys[:,1:T-1] .- repeat(mean(ys, dims = 2),1, T-1)))
    end

    if flag == 1
        return bhat, MTHs[use], zs[:,:,use]
    elseif flag == 2
        return bhat, MTHs
    end
end

function get_STH(iT::Int64, use::Array{Int64, 1}, MT::Array{Float64, 1}, e_sim::Array{Float64, 2}, H::Int64 = 10, T::Int64 = 200)

    bhat0, MTH0, z0 = get_b(1, Matrix(1.0I,length(use),length(use)), MT, use, e_sim, H, T)

    Γs = zeros(length(use), length(use), iT+1)
    for i in 1:(iT+1)
        for j in 1:H, k in i:T
            Γs[:,:,i] += reshape(z0[j,k,:] .- MTH0, length(use), 1) * reshape(z0[j,k-i+1,:] .- MTH0, length(use), 1)'
        end
    end
    Γs = Γs ./ (T*H)

    STH = Γs[:,:,1]
    for i in 2:(iT+1)
        STH += (1 - (i-1) / (iT+1)) .* (Γs[:,:,i] .+ Γs[:,:,i]')
    end
    STH = (1 + 1/H) .* STH

    return bhat0, STH
end

function get_derivative(bhat::Array{Float64,1}, MTH::Array{Float64, 1}, H::Int64, T::Int64, esim::Array{Float64, 2}, use::Array{Int64, 1})
    MTH0 = MTH

    ys_rho = zeros(H,T+1)
    ys_sigma = zeros(H,T+1)

    MTs_rho = zeros(H, 3)
    MTs_sigma = zeros(H, 3)
    for i in 1:H
        for t in 2:T+1
            ys_rho[i,t] = (bhat[1] - 0.001) .* ys_rho[i,t-1] .+ bhat[2] .* e_sim[i,t-1]
            ys_sigma[i,t] = bhat[1].* ys_sigma[i,t-1] .+ (bhat[2] - 0.001) .* e_sim[i,t-1]
        end
        MTs_rho[i,:] = get_MT(ys_rho[i,2:T+1])
        MTs_sigma[i,:] = get_MT(ys_sigma[i,2:T+1])
    end
    MTH_rho = [mean(MTs_rho[:,1]), mean(MTs_rho[:,2]), mean(MTs_rho[:,3])]
    MTH_sigma = [mean(MTs_sigma[:,1]), mean(MTs_sigma[:,2]), mean(MTs_sigma[:,3])]

    g1 = (MTH0 .- MTH_rho) ./ 0.001
    g2 = (MTH0 .- MTH_sigma) ./ 0.001
    g = hcat(g1[use], g2[use])
    return g
end

function get_cov(STH::Array{Float64, 2}, g::Array{Float64, 2}, T::Int64)
    cov = inv(g' * inv(STH) * g) ./ T
    std = sqrt.(diagm(diag(cov)))
    return cov, std
end

function jtest(bhat::Array{Float64, 1}, W::Array{Float64, 2}, MT::Array{Float64, 1}, H::Int64, T::Int64, use::Array{Int64, 1})
    Js = get_J(W, MT, use, H, T, bhat[1], bhat[2])

    ts = (T * H / (1+H)) * Js

    return ts
end

#### Problem Set

## Fix the true data and error terms

xs = gen_true()
MT = get_MT(xs)
e_sim = gen_sim()

## SMM using mean and variance as the moments

bhat1_0, STH1 = get_STH(4, [1,2], MT, e_sim, 10, 200)
bhat1_1, MTHs1 = get_b(2,inv(STH1), MT, [1,2], e_sim, 10, 200)
gbhat1 = get_derivative(bhat1_1, MTHs1, 10, 200, e_sim, [1,2])
cov1, std1 = get_cov(STH1, gbhat1, 200)
js1 = jtest(bhat1, inv(STH1), MT, 10, 200, [1,2])

## SMM using variance and autocorrelation

bhat2_0, STH2 = get_STH(4, [2,3], MT, e_sim, 10, 200)
bhat2_1, MTHs2 = get_b(2, inv(STH2), MT, [2,3], e_sim, 10, 200)
gbhat2 = get_derivative(bhat2_1, MTHs1, 10, 200, e_sim, [2,3])
cov2, std2 = get_cov(STH2, gbhat2, 200)
js2 = jtest(bhat2, inv(STH2), MT, 10, 200, [2,3])

## Bootstrap exercise

function get_bs(B::Int64)
    bhat_bs = zeros(B,2,2)

    for i in 1:B
        xs = gen_true()
        MT = get_MT(xs)
        e_sim = gen_sim()

        bhat_1, STH = get_STH(4, [1,2,3], MT, e_sim, 10, 200)
        bhat_2, MTHs = get_b(2, inv(STH), MT, [1,2,3], e_sim, 10, 200)

        bhat_bs[i,:,1] = bhat_1
        bhat_bs[i,:,2] = bhat_2
    end
    return bhat_bs
end

get_bs(10000)
