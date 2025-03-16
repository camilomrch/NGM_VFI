############################################################################################
# This code solves the household's problem using the value function iteration (VFI) method.
# The solution is obtained by continuous optimization.
# Stochastic growth model.
# Programmer: Camilo Marchesini, February 9, 2025.
#############################################################################################

# To activate the environment, run the following commands once in the terminal, then comment out.
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
        kgrid::Vector{Float64}            = collect(range(0.25*( α / ((1/β)-(1-δ))  )^(1 / (1 - α)), 1.75*( α / ((1/β)-(1-δ))  )^(1 / (1 - α)), nk))
        # Discretize productivity process.
        ρ::Real                           = 0.966
        sigm::Real                        = 0.5
        nA::Int                           = 7 
        # Maximum number of iterations.
        maxiter::Int                      = 1000
        # Tolerance criterion.
        tol::Float64                      = 0.01 
        
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




  
function rouwenhorst_Pi(N::Int, p::Real)
    
        # Base case: Pi^2.
        Pi = [p 1 - p;
              1 - p p]
        # Recursion to build Pi^n up from Pi^2 to Pi^N.
        for n in 3:N
            Pi_old = Pi
            Pi = zeros(n, n)
            
            # Update Pi^n using the recursion.
            Pi[1:end-1, 1:end-1] += p * Pi_old
            Pi[1:end-1, 2:end] += (1 - p) * Pi_old
            Pi[2:end, 1:end-1] += (1 - p) * Pi_old
            Pi[2:end, 2:end] += p * Pi_old
            
            # Normalize all rows but the first and last rows.
            Pi[2:end-1, :] ./= 2
        end
        
        return Pi
end



function stationary_markov(Pi::Matrix{T}, tol::T=1e-14, max_iter::Int=10_000) where T<:Real # Enforce that type T must be a subtype of Real.

        # Initialize flag.
        noconvergence = true; 
        # Initialize iteration counter.
        iter = 0
    
        # Initialize using a uniform distribution over all states.
        n = size(Pi, 1)
        pi = fill(1/n, n)
    
        # Update distribution using Pi until successive iterations differ by less than the tolerance.
        while noconvergence && iter < max_iter
    
            # Move to next iteration.
            iter += 1
    
            # Updating step.
            pi_new = Pi' * pi
    
            # Check for convergence.
            if maximum(abs.(pi_new - pi)) < tol
                # Switch flag to false.
                noconvergence = false; 
                #println("Distribution of Markov chain converged in $iter iterations.")
                return pi_new # return the stationary distribution.
            end
            # Prepare for next updating step.
            pi = pi_new
        end
    
        # If the maximum number of iterations is reached without convergence, inform the user and return nothing.
        if noconvergence && iter >= max_iter
            println("Distribution of Markov chain did not converge after $max_iter iterations!")
            return nothing
        end
        
        return pi  
end
    
    


function discretize_income(ρ::Real, σ::Real, n_e::Int)
    
        # Choose inner-switching probability p to match persistence ρ.
        p = (1 + ρ) / 2
        
        # Initialize using states 0 through n_e-1 and scale by α to match standard deviation σ.
        e = collect(0:n_e-1)
        α = 2 * σ / sqrt(n_e - 1)
        e = α .* e
        
        # Obtain Markov transition matrix Pi and its stationary distribution.
        Pi = rouwenhorst_Pi(n_e, p)
        pi = stationary_markov(Pi)
        
        # Since e is log income, get income y by exponentiating.
        y = exp.(e)
    
        # Normalization: divide each element of y by the scalar dot(pi, y), such that 
        # the expectation of y equals 1. 
        y /= dot(pi, y)
        
        return y, pi, Pi
end
    


function TV(kp,k,A,V,Π_ishock,Agrid,paramss::ModelParameters)

      @unpack_ModelParameters paramss
      # Interpolate value function.
      Vp_interp = [interp(kp, kgrid, V[:, shock_idx]) for shock_idx in eachindex(Agrid)]
      c              = A*F(k) + (1 - δ)*k - kp

      if c < 0.0
         val = -1e6 - 100 * abs(c)
      else
         val = u(c) + β * dot(Vp_interp, Π_ishock)
      end

  return -val 
end




function VFI_continuous_optim(paramss::ModelParameters)
        
       # Unpack model parameters.
       @unpack_ModelParameters paramss


       Agrid,_,Π = discretize_income(ρ, sigm, nA)
       
       # Initial guess for the value function.
	   V               = zeros(nk,nA)
       ## Pre-allocate.
       # Updated value function after iteration step.
       V_new           = zeros(nk,nA)
       kpstar          = zeros(nk,nA)
     
       # Initialize iteration counter.
       iter            = 0

      # Initialize convergence criterion.
       maxdiff         = 10


        while (maxdiff > tol && iter < maxiter)
        
            for (ishock,A) in enumerate(Agrid)
              for (ik,k) in enumerate(kgrid)
            

                   res = optimize(kp -> TV(kp, k, A, V,Π[ishock,:],Agrid, paramss), minimum(kgrid), maximum(kgrid), Brent())
            
                   V_new[ik,ishock]                = -Optim.minimum(res) 
                   kpstar[ik,ishock]               = Optim.minimizer(res)

               end
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


# Discretize income process.
Agrid,_,Π = discretize_income(mp.ρ, mp.sigm, mp.nA)


# Initialize plot for looping through.
pv = plot()

# Loop over each value in y to plot the consumption function.
for (ishock, Ai) in enumerate(Agrid)
    plot!(pv,mp.kgrid, V[:, ishock], label = "A=$(round(Ai, digits=2))", xlabel="Current capital, k", ylabel="Value function, v", linewidth=2)
end
display(pv)

pkp = plot()

# Loop over each value in y to plot the consumption function.
for (ishock, Ai) in enumerate(Agrid)
    plot!(pkp,mp.kgrid, kpstar[:, ishock], label = "A=$(round(Ai, digits=2))", xlabel="Current capital, k", ylabel="Policy function, k'", linewidth=2)
end
plot!(mp.kgrid, mp.kgrid,  primary=false, linestyle=:dash, linewidth=2, color=:black)
display(pkp)


