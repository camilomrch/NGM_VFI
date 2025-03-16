####################################################################################################
# This code solves the household's problem using the value function iteration (VFI) method.
# The solution is obtained by discrete optimization (grid search), not by continuous optimization.

# Programmer: Camilo Marchesini, February 9, 2025.
####################################################################################################

# To activate the environment, run the following commands once in the terminal.
import Pkg
Pkg.activate(@__DIR__) 
Pkg.instantiate()

# Load packages.
using LinearAlgebra
using Plots
using Parameters

# Structure to store model parameters, including structural 
# and numerical ones.
@with_kw mutable struct ModelParameters

        #########################
        ## Household parameters.
        #########################

        σ :: Float64 = 2.0
        β :: Float64 = 0.95
        α :: Float64 = 0.33
        δ :: Float64 = 0.1

        #########################
        ## Numerical parameters.
        #########################

        # Number of nodes of capital grid.
	     nk::Int 	                    = 100 
        # Grid of capital.
        kgrid::Vector{Float64}         = collect(range(0.25*( α / ((1/β)-(1-δ))  )^(1 / (1 - α)), 1.75*( α / ((1/β)-(1-δ))  )^(1 / (1 - α)), nk))
        # Maximum number of iterations.
        maxiter::Int                  = 1000
        # Tolerance criterion.
        tol::Float64                  = 1e-6 
        # Initialize distance metric.
        maxdiff::Float64              = 10
        
        ##################
        # Model equations.
        ##################
        
        # Utility function.
        u::Function = c -> c^(1 - σ)/(1 - σ)
        # Production function.
        F::Function = k -> k^α
        
end


"""
    VFI_discrete_optim(paramss::ModelParameters)

Solves the household's problem using the value function iteration (VFI) method. Optimization is discrete, via grid search.

# Example.
```jldoctest
julia> params = ModelParameters()
@time begin
(V, pol, iterations) =  VFI_discrete_optim(para)
end
```

"""
function VFI_discrete_optim(paramss::ModelParameters)
        
       # Unpack model parameters.
       @unpack_ModelParameters paramss
       
       # Initial guess for the value function.
	    V               = zeros(nk)
       ## Pre-allocate.
       # Updated value function after iteration step.
       V_new           = similar(V) 
       # Vector of indices of the argmax's ((k')^*).
       pol             = similar(V, Int) 
       # Consumption policy function c(k,k').
       c               = similar(V, (nk,nk))
       # Contraction mapping.
       TV              = similar(c)
       # Boolean mask for positive consumption.
       positive_c_mask = similar(c, Bool)

       # Pre-compute consumption policy function c(k,k').
       c              .= F.(kgrid) .+ (1 - δ).* kgrid .- kgrid'

       # Initialize iteration counter.
       iter            = 0


        while (maxdiff > tol && iter < maxiter)
           # Set TV to -Inf, 
           # then populate by applying Bellman operator for states where consumption is positive.
           TV                       .= -Inf 
           positive_c_mask          .= (c .> 0.0)
           # Apply Bellman operator T.
           TV[positive_c_mask]      .= u.(c[positive_c_mask]) .+ β .*(repeat(V, outer = [1,nk]))'[positive_c_mask]
           # Find the maximum value function and the corresponding policy function ((k')^*).
           Vmax, argmax_pos          = findmax(TV, dims=2)
           # Vectorize and assign updated value function.
           V_new                    .= vec(Vmax)
           # Store the column indices, corresponding to the (k')^*'s.
           pol                      .= getindex.(argmax_pos, 2)
           # Evaluate convergence criterion.
           maxdiff                   = maximum(abs.(V_new .- V))
           # Test convergence conditions.
              if maxdiff < tol
                 break
              end
              if iter >= maxiter
                 println("Cannot solve constrained household's problem: No convergence after $maxiter iterations!")
                 break
              end
           # Re-assign updated value function to the current value function in view of new iteration.
           V                        .= V_new

           # Move to next iteration.
           iter                     += 1
        end

    return  (V=V, pol=pol, iter=iter)
end

# Instantiate structure.
mp = ModelParameters()
# Evaluate function, timing execution.
@time begin 
   (V, pol, iterations) =  VFI_discrete_optim(mp)
end
# Print number of iterations to REPL.
println("The value function converged in $iterations loops.")

pltv  = plot(mp.kgrid, V, primary=false, xlabel = "Current capital, k ", ylabel = "Value function, v", linewidth= 2, color = :blue)
pltkp = plot(mp.kgrid, mp.kgrid[pol], primary=false, xlabel = "Current capital stock, k ", ylabel = "Policy function, k'", linewidth= 2, color = :blue)
plot!(mp.kgrid, mp.kgrid,  primary=false, linestyle=:dash, linewidth=2, color=:black)
