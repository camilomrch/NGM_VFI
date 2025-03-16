############################################################################################
# This code solves the household's problem using the value function iteration (VFI) method.
# The solution is obtained by continuous optimization.

# Programmer: Camilo Marchesini, February 9, 2025.
#############################################################################################

# To activate the environment, run the following commands once in the terminal.
import Pkg
Pkg.activate(@__DIR__) 
Pkg.instantiate()

# Load packages.
using LinearAlgebra
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
        β :: Float64 = 0.95
        α :: Float64 = 0.33
        δ :: Float64 = 0.1

        #########################
        ## Numerical parameters.
        #########################

        # Number of nodes of capital grid.
	     nk::Int 	                      = 100 
        # Grid of capital.
        kgrid::Vector{Float64}         = collect(range(0.25*( α / ((1/β)-(1-δ))  )^(1 / (1 - α)), 1.75*( α / ((1/β)-(1-δ))  )^(1 / (1 - α)), nk))
        # Maximum number of iterations.
        maxiter::Int                    = 1000
        # Tolerance criterion.
        tol::Float64                    = 0.01 
        
        ##################
        # Model equations.
        ##################
        
        # Utility function.
        u::Function = c -> c^(1 - σ)/(1 - σ)
        # Production function.
        F::Function = k -> k^α
        
end


function interp(x, xp, fp)

   # Handle extrapolation by imposing bounds.
   if x <= xp[1]
      # Left extrapolation.
       return fp[1]  
   elseif x >= xp[end]
       # Right extrapolation.
       return fp[end] 
   end

   # Find interval on xp within which x falls.
   j = searchsortedfirst(xp, x)

   # Interpolate. 
   t = (x - xp[j-1]) / (xp[j] - xp[j-1])
   return fp[j-1] + t * (fp[j] - fp[j-1])
end


function TV(kp,k,V,paramss::ModelParameters)

      @unpack_ModelParameters paramss

       Vp_interp      = interp(kp,kgrid,V)
       c              = F(k) + (1 - δ)*k - kp

      if c < 0.0
         val = -1e6 - 100 * abs(c)
      else
         val = u(c) + β * Vp_interp
      end

  return -val 
end




function VFI_continuous_optim(paramss::ModelParameters)
        
       # Unpack model parameters.
       @unpack_ModelParameters paramss
       
       # Initial guess for the value function.
	    V               = zeros(nk)
       ## Pre-allocate.
       # Updated value function after iteration step.
       V_new           = zeros(nk)
       kpstar          = zeros(nk)
     
       # Initialize iteration counter.
       iter            = 0

      # Initialize convergence criterion.
       maxdiff         = 10


        while (maxdiff > tol && iter < maxiter)
           for (ii,k) in enumerate(kgrid)
            
            res = optimize(kp -> TV(kp, k, V, paramss), minimum(kgrid), maximum(kgrid), Brent())
            
            V_new[ii]                = -Optim.minimum(res) 
            kpstar[ii]               = Optim.minimizer(res)

           end
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

    return  (V=V, kpstar=kpstar, iter=iter)
end

# Instantiate structure.
mp = ModelParameters()
# Evaluate function, timing execution.
@time begin 
   (V, kpstar, iterations) =  VFI_continuous_optim(mp) 
end
# Print number of iterations to REPL.
println("The value function converged in $iterations loops.")

pltv  = plot(mp.kgrid, V, primary=false, xlabel = "Current capital, k ", ylabel = "Value function, v", linewidth= 2, color = :blue)
pltkp = plot(mp.kgrid, kpstar, primary=false, xlabel = "Current capital stock, k ", ylabel = "Policy function, k'", linewidth= 2, color = :blue)
plot!(mp.kgrid, mp.kgrid,  primary=false, linestyle=:dash, linewidth=2, color=:black)
