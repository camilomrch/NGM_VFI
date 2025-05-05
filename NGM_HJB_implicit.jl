#########################################################################################
# NGM in continuous time using HJB approximated 
# with finite difference method, implicit scheme.
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





function solve_HJB_implicit(paramss::ModelParameters)

    @unpack_ModelParameters paramss

    kmin = kgrid[1]
    kmax = kgrid[end]

    # Grid step size.
    Δk  = (kgrid[end] - kgrid[1])/(nk - 1)
    # Time step size (its size does not matter).
    Δt = 1000 

    # Preallocate arrays.
    dVf                 = zeros(nk)
    dVb                 = zeros(nk)
    dV0                 = zeros(nk)
    dV_upwind           = zeros(nk)
    Vchange             = zeros(nk)
    c                   = zeros(nk)
    cf                  = zeros(nk)
    cb                  = zeros(nk)
    c0                  = zeros(nk)
    muf                 = zeros(nk)
    mub                 = zeros(nk)
    X                   = zeros(nk)
    Y                   = zeros(nk)
    Z                   = zeros(nk)

    b                   = zeros(nk)

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
        dVf[nk] = (F.(kmax) - δ .* kmax)^(-σ)
        dVb[2:nk]   .= (V[2:nk] - V[1:nk-1]) / Δk
        dVb[1] = (F.(kmin) - δ .* kmin)^(-σ)

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

        # Use relevant difference based on indicators.
        dV_upwind .= dVf.*If .+ dVb.*Ib .+ dV0.* I0
        # Compute consumption from FOC.
        c .= dV_upwind.^(-1/σ)

        #############################################
        # Construct infinitesimal generator matrix A.
        #############################################

        # Lower diagonal elements.
        X = -min.(mub, 0)/Δk
        # Main diagonal elements.
        Y = -max.(muf, 0)/Δk + min.(mub, 0)/Δk
        # Upper diagonal elements.
        Z =  max.(muf, 0)/Δk

        # Construct sparse matrix A.
        A = spdiagm(
            -1 => X[2:end],      # Lower diagonal (nk - 1).
             0 => Y,             # Main diagonal (nk).
             1 => Z[1:end-1]     # Upper diagonal (nk - 1).
        )

        # Construct sparse matrix B.
        B = (ρ + 1/Δt) * spdiagm(0 => ones(nk)) - A
        b .= u.(c) .+ V ./ Δt

        # Solve linear system.
        V .= B \ b

        # Compute change in the value function.
        Vchange .= V .- v

        # Evaluate convergence criterion.
        maxdiff                      = maximum(abs.(Vchange))

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

        # Update value function.
         v       .= V

        # Move to next iteration.
        iter                     += 1

    end

end



####################
# IMPLICIT METHOD.
####################

# Instantiate structure (using many gridpoints).
mpp = ModelParameters(nk=10000)
# Evaluate function, timing execution.
@time begin
    # Solve HJB equation.
    (v, c, iterations) = solve_HJB_implicit(mpp) 
end

# Savings decision function.
kdot = mpp.F.(mpp.kgrid) - mpp.δ .* mpp.kgrid - c

# Plot savings decision.
plt_kdot = plot(mpp.kgrid, kdot, xlabel="k", ylabel="s(k)", title="Optimal savings policy",primary=false)
plot!(mpp.kgrid, zeros(length(mpp.kgrid)),  primary=false, linestyle=:dash, linewidth=2, color=:black)
