#########################################################################################
# NGM in continuous time using HJB approximated 
# with finite difference method, explicit scheme.
# Based on Matlab code by Benjamin Moll, but see appendix of Achdou et al. (2022)
# for more details.

# Programmer: Camilo Marchesini, May 3, 2025.
#########################################################################################


#import Pkg
#Pkg.activate(@__DIR__) 
#Pkg.instantiate()

# Load packages.
using LinearAlgebra
using SparseArrays
using Plots
using Parameters
using Optim


# Structure to store model parameters, including structural 
# and numerical ones.
@with_kw mutable struct ModelParameters

    #########################
    ## Household parameters.
    #########################

    σ :: Float64 = 2.0
    ρ :: Float64 = 0.05
    α :: Float64 = 0.3
    δ :: Float64 = 0.05

    #########################
    ## Numerical parameters.
    #########################

    # Number of nodes of capital grid.
    nk::Int 	                      = 150
    # Grid of capital.
    kgrid::Vector{Float64}            = collect(range(0.001*( α / (ρ+δ)  )^(1 / (1 - α)), 2*( α / (ρ+δ)  )^(1 / (1 - α)), nk))
    # Maximum number of iterations.
    maxiter::Int                      = 10000
    # Tolerance criterion.
    tol::Float64                      = 10e-6
    
    ##################
    # Model equations.
    ##################
    
    # Utility function.
    u::Function = c -> c^(1 - σ)/(1 - σ)
    # Production function.
    F::Function = k -> k^α
    
end


function solve_HJB_explicit(paramss::ModelParameters)

    @unpack_ModelParameters paramss


    # Grid step size.
    Δk  = (kgrid[end] - kgrid[1])/(nk - 1)
    # Time step size (small enough to ensure the Courant-Friederichs-Lewy numerical stability condition is satisfied).
    Δt = 0.9 * Δk/ (maximum(F.(kgrid) - δ .* kgrid))

    # Preallocate arrays.
    dVf                 = zeros(nk)
    dVb                 = zeros(nk)
    dV0                 = zeros(nk)
    dV_upwind           = zeros(nk)
    Vchangeoverdeltat   = zeros(nk)
    c                   = zeros(nk)
    cf                  = zeros(nk)
    cb                  = zeros(nk)
    c0                  = zeros(nk)
    muf                 = zeros(nk)
    mub                 = zeros(nk)

    # Elicit initial guess for value function.
    v0                  = u.(F.(kgrid)) / ρ
    # Feed guess to loop below to initialize it.
    v                   = copy(v0)


    # Initialize convergence criterion.
    maxdiff             = 100
    # Start iteration counter.
    iter                = 0
    # Start loop.
    while (maxdiff > tol && iter < maxiter)
        
        # Assign new value function guess.
        V = copy(v)

        # Approximating V' by computing forward and backward differences of V. 
        dVf[1:nk-1] .= (V[2:nk] - V[1:nk-1]) / Δk
        dVf[nk] = 0;
        dVb[2:nk]   .= (V[2:nk] - V[1:nk-1]) / Δk
        dVb[1] = 0;

        # Computing optimal consumption and drift, based on forward and backward differences.
        cf .= dVf.^(-1/σ)
        muf .= F.(kgrid) - δ .* kgrid - cf
        cb .= dVb.^(-1/σ)
        mub .= F.(kgrid) - δ .* kgrid - cb

        # Steady-state consumption (from optimal drift equal zero) 
        # and steady-state V' (from first-order condition evaluated in steady state).
        c0  .= F.(kgrid) - δ .* kgrid
        dV0 .= c0.^(-σ)

        ## Upwind scheme: 
        ## use forward difference whenever drift of state variable positive, backward difference whenever drift of state variable negative.
        
        # Computer indicators.
        If = muf .> 0
        Ib = mub .< 0
        I0 = .~(If .| Ib)
        # make sure the right approximations are used at the boundaries.
        Ib[1] = 0; If[1]  = 1; Ib[nk] = 1; If[nk] = 0;

        # Use relevant difference based on indicators.
        dV_upwind .= dVf.*If .+ dVb.*Ib .+ dV0.* I0
        
        # Compute consumption from FOC.
        c .= dV_upwind.^(-1/σ)
        # Compute derivative of the value function with respect to time.
        Vchangeoverdeltat .= u.(c) + dV_upwind.*(F.(kgrid) - δ .* kgrid - c) - ρ.*V;

        # Update value function.
        v .= v .+ Δt.* Vchangeoverdeltat

        # Evaluate convergence criterion.
        maxdiff                                = maximum(abs.(Vchangeoverdeltat))

        # Check for convergence.
        if maxdiff < tol
            println("Convergence reached after $iter iterations.")
            return  (v=v, c=c, iter=iter)
            break
        end
        if iter >= maxiter
            println("Cannot solve constrained household's problem: No convergence after $maxiter iterations!")
            break
         end


        # Move to next iteration.
        iter                     += 1

    end

end
    

####################
# EXPLICIT METHOD.
####################

# Instantiate structure.
mp = ModelParameters()
# Evaluate function, timing execution.

@time begin
    # Solve HJB equation.
    (v, c, iterations) = solve_HJB_explicit(mp) 
end

kdot = mp.F.(mp.kgrid) - mp.δ .* mp.kgrid - c

# Compute and plot also optimal k' for comparison with the other methods I use in the other scripts in this folder.
kpstar = mp.F.(mp.kgrid) + (1-mp.δ) .* mp.kgrid - c
plt_kpstar = plot(mp.kgrid, kpstar, xlabel="k", ylabel="k'", title="Optimal choice of capital", primary=false)


# Plot savings decision.
plt_kdot = plot(mp.kgrid, kdot, xlabel="k", ylabel="s(k)", title="Optimal savings policy",primary=false)
plot!(mp.kgrid, zeros(length(mp.kgrid)),  primary=false, linestyle=:dash, linewidth=2, color=:black)

