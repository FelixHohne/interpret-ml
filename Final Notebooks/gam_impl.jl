using LinearAlgebra
using RDatasets
using GLM
using Pkg
using CSV
using DataFrames
using Plots
using Pkg
using Distributions

function spectral_sqrt(X)
    e, U = eigen(X) 
    spectral_root =  U * Diagonal(e .^0.5) * U'
    return spectral_root
end

"""
Compute cubic spline basis
x: single data point 
z: single knot location
Documentation for outer product computation:
https://stackoverflow.com/questions/44591481/julia-outer-product-function
"""
function rk(x, z) 
   return ((z-0.5)^2-1/12)*((x-0.5)^2-1/12)/4-
   ((abs(x-z)-0.5)^4-(abs(x-z)-0.5)^2/2+7/240)/24
end     

"""
    x_k: knot locations
    x: data vector
    return X: augmented data matrix for specific vector with bases computed
"""
function constructAugX(x, xk) 
    q = length(xk) + 2
    n = length(x)
    X = ones(n, q)
    X[:, 2] = x
    X[:, 3:q] = rk.(x, xk') 
    return X
end;

struct Scaler
    max_values::Vector{Float64}
    min_values::Vector{Float64}
end



"""
Construct the S wiggliness penalization matrix
x_k: knot locations
p: number of covariates 
i: Computing the S_i penalty
"""
function constructS(xk) 
    q = length(xk) + 2
    S = zeros(q, q) 
    # First two rows and columns of S are 0 by construction 
    # S_{i+2}, S_{j+2} = R(x_{i}^*, x_{j}^{*}) by construction
    S[3:end, 3:end] = rk.(xk, xk')
    return S
end 



"""
sp: penalization factors
q: number of knots
"""
function construct_augmented(X, chosen_knots=nothing, q=10) 
    n, p = size(X)

    if !isnothing(chosen_knots)
        knot_locs = chosen_knots 
        q = size(knot_locs)[1] + 2
    else 
        knot_locs = zeros(q-2, p) # Each column contains knot locations for one covariate
        for k in 1:p
        
            knot_locs[:, k] = quantile(unique(X[:, k]), (1:q-2)/(q-1))
        end
    end
    
        
    S_list = []
    current_pos = 0 
    for k in collect(1:p)
        S_k = zeros(p * q - (p - 1), p * q - (p - 1))
       
        if k == 1
            S_k[1:q, 1:q] = constructS(knot_locs[:, k])
            current_pos = q
        else
            S_k[current_pos + 1:(current_pos + q -1), current_pos + 1: (current_pos + q-1)]  = constructS(knot_locs[:, k])[2:end, 2:end]
            current_pos = current_pos + q-1
        end
        push!(S_list, S_k)
    end 
    
    augX_list = []   
    for k in 1:p
        computed_X = constructAugX(X[:, k], knot_locs[:, k])
        if k == 1
            push!(augX_list, computed_X)
        else 
            push!(augX_list, computed_X[:, 2:end]) # Only push intercept term once 
        end
    end
    augX = reduce(hcat, augX_list)   
    
    return knot_locs, augX, S_list

end;

function generate_penalized(augX, S_list, sp)
    @assert length(sp) == length(S_list) 
    rS = sp[1] .* S_list[1]
    for i in 2:length(S_list)
        rS += sp[i] .* S_list[i]
    end
    rB = spectral_sqrt(rS) 

    penalizedX = vcat(augX, rB)
    return penalizedX, rS
end

function compute_gcv(y, preds)
    n = length(y)
    rss = sum((y - preds).^2)
    trI_A = 0 
    return n * rss / (tr(I_AA)^2)
end

# GAM Fitting code     
function fitGAM(y, augX, S_list, sp) 
    n, p = size(augX)
    
    penalizedX, rS = generate_penalized(augX, S_list, sp)
    
    
    b = zeros(p)
    b[1] = 1
    
    num_iters = 10
    
    change_norm = 1000000
    
    for i in 1:num_iters
        eta = (penalizedX * b)[1:n]
        @assert !any(isnan, eta) "$i"
        mu = exp.(eta)
        @assert !any(isnan, mu) "$i"

        z_orig = ((y - mu) ./ mu) + eta
        @assert !any(isnan, z_orig) " eta: $eta"

        z = vcat(z_orig, zeros(p))  
        @assert !any(isnan, z) "$i"

        new_beta = penalizedX \ z
        @assert !any(isnan, new_beta) "$i"

        change_norm = norm(new_beta - b)
        # println(change_norm)
        b = new_beta
        if change_norm < 1e-8
            break
        end

        if i == num_iters
            # v = X' * z_orig
            # mean = (X' * X + rS) \ v
            # if all(x-> x == 0, sp)
            #     covariance = inv(penalizedX' * penalizedX) # ϕ = 1
            # else
            #     covariance = inv(penalizedX' * penalizedX + rS) # ϕ = 1
            # end

            l = size(penalizedX)[1]
            covariance = inv(penalizedX' * Diagonal(ones(l)) * penalizedX + rS)
            std_vector = sqrt.(diag(covariance))

            @assert size(b) == size(std_vector)




            return b, (b - 1.96 * std_vector), (b + 1.96 * std_vector)
            
        end 


    end 
    return b, beta_CI
end 

function encode_gender(gender)
    if gender == "M"
        return 0
    elseif gender == "F"
        return 1
    end
end;
