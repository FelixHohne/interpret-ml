using LinearAlgebra
using RDatasets
using GLM
using Pkg
using CSV
using DataFrames
using Plots
using Pkg
using Distributions
using StatsBase

function _wls_solve(w, F, b) 
    # Computes min_x ||WAX - b||_2^2 assuming W diagonal Matrix represented as vector
    # F is the SVD representation of A matrix. 

    z = F.U' * b   
    y = z ./ (w[1:length(F.S)] .* F.S)
    return F.V * y 
end;

function _phi_estimator(y, X, w_k, mu_hat)
    # Implements the Pearson Estimator of ϕ
    
    n, p = size(X)
    pearson_stat = sum(w_k .* (((y - mu_hat).^2) ./ mu_hat.^2))
    return pearson_stat / (n - p)
end


function _pred_intervals(g_inverse, X, w_k, beta_k, ϕ)

    n, _ = size(X)
    _X = [ones(n) X]

    v = sqrt.(diag(inv(_X' * Diagonal(w_k) * _X)))
    beta_se = sqrt(ϕ) .* v
    beta_lower = beta_k - 1.96 .* beta_se
    beta_upper = beta_k + 1.96 * beta_se 

    μ_lower = g_inverse.(_X * beta_lower)
    μ_lower = g_inverse.(_X * beta_upper)

    return μ_lower, μ_lower
end

function _irls(X, y, assume_stable=false, method="poisson",return_phi = false, test_beta=Float64[])
    # set assume_stable = true if problem is well-conditioned. 
    # assume_stable = true will solve via LU decomposition 
    # Otherwise, use stable but slower SVD decomposition
    
    # Requires: method be `poisson` or `gamma`
    
    m, n = size(X)
    @assert m >= n "GLMs require n > p"
    X = hcat(ones(m), X)
  
    # if method == "poisson"
    #     β_k = zeros(n+1) 
    # else 
    #     β_k = zeros(n+1) .+ 0.1
    # end 

    β_k = zeros(n+1) .- 0.00001
    cur_diff = 1e32
    ϵ_machine = 1e-8
  
    if assume_stable == false
        F = svd(X)
    end
   
    # Does not affect estimation of $\hat{\beta}$
    ϕ = 0.5
    # g = x -> log(x)
    if method == "poisson"
        g_prime = x -> 1 / x
        g_inverse = x -> exp(x)
    else
        g_prime = x -> 1 / (x^2)
        g_inverse = x -> -1 / x
    end 
       
    
    μ_k = g_inverse.(X * β_k)  
    @assert !any(isnan, μ_k) "Failed at beginning"
    w_k = 1 ./ (g_prime.(μ_k) .* ϕ)
    @assert !any(isnan, w_k) "Failed at beginning"
    
    differences = []
    
    iter_count = 0
    while cur_diff > ϵ_machine 
    # for i in 1:200
        μ_k = g_inverse.(X * β_k)  
        @assert !any(isnan, μ_k) "$i"
        w_k = 1 ./ (g_prime.(μ_k) .* ϕ)
        @assert !any(isnan, w_k) "$i"
        
        if assume_stable
            W_k = Diagonal(w_k)
        end
        
        y_primek = g_prime.(μ_k) .* y 
        @assert !any(isnan, y_primek) "$i"
        μ_primek = g_prime.(μ_k) .* μ_k
        @assert !any(isnan, μ_primek) "$i"

        iter_count += 1

        # Naively: new_β = β_k + inv(X' * W_k * X) * X' * W_k * (y_primek - μ_primek)
        if assume_stable
            new_β = β_k + (X' * W_k * X) \ X' * (W_k * (y_primek - μ_primek))
        else
            new_β = β_k + _wls_solve(w_k, F, Diagonal(w_k) * (y_primek - μ_primek))
        end 
        
        cur_diff = norm(new_β - β_k)
        push!(differences, cur_diff)
        β_k = new_β
    end 
    
    # Implement Wald test
    if length(test_beta) > 0 || return_phi == true
        # Fisher Information 
        ϕ = _phi_estimator(y, X, w_k, μ_k)
        println("ϕ_estimate is: ",  ϕ)

        if length(test_beta) > 0
            I_info = X' * Diagonal(w_k) * X 
            # m used here, as it is number of examples
            W = m * (-β_k - test_beta)' * I_info * (-β_k - test_beta) 
            # W converges in distribution to chi_squared with df = p (need to count intercept)  
            # Compute qchisq(df=p) in R
            test_value = cquantile(Chisq(n + 1), alpha)
            if W > test_value
                println("test beta is rejected at alpha = 0.05")
            else
                println("test beta is not rejected at alpha = 0.05")
            end
    end 

    if return_phi 
        return differences, -β_k, ϕ, w_k
    else
        return differences, -β_k
    end 
  end
end 