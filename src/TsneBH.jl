module TsneBH

export tsne

using LinearAlgebra
using Statistics

function tsne(X::Matrix{Float64},emb_size::Int64,T::Int64;
                lr::Float64=1.,perp::Float64=30.,tol::Float64=1e-5,
                max_iter::Int=50,pca::Bool=true,pca_dim::Int=50,verbose::Bool=true)
    # Create an initial random embedding
    Y = randn(size(X)[1],emb_size)
    # Perform PCA if selected
    pca && (X = PCA(X,pca_dim))
    # Search for the sigmas of the Gaussians
    P, sigma = binary_search(X,perp,tol,max_iter,v=verbose)
    # Start the iteration
    Y_new = gradient_descent(T,P,Y,lr,v=true)
end

function gradient_descent(T::Int64,P::Matrix{Float64},data_red::Matrix{Float64},lr::Float64;v::Bool)
    # Start iterating
    dr = copy(data_red)
    println("Beginning gradient descent...")
    for t in 1:T
        # Compute gradient
        q_distr, num = cond_t(dr)
        size(P) != size(q_distr) && throw(ArgumentError("sizes don't match"))
        dy = grad_KL(P,dr,q_distr,num)
        # Update values
        dr = dr - lr * dy
        # Compute loss
        if v && t % 10 == 0
            println("t=$t, C=$(cost_KL(P,q_distr))")
        end
        # Center in zero
        dr = dr - repeat(sum(dr,dims=1)/size(dr)[1],size(dr)[1],1)
    end
    dr
end

function grad_KL(P::Matrix{Float64},data_red::Matrix{Float64},q_distr::Matrix{Float64},num::Matrix{Float64})
    PQ = P - q_distr
    dy = zeros(size(data_red))
    for i in 1:size(dy)[1]
        dy[i,:] = sum(repeat(PQ[:,i] .* num[:,i],1,size(data_red)[2]) .* (transpose(data_red[i,:]) .- data_red),dims=1)
    end
    dy
end

function PCA(data::Matrix{Float64},dims::Int)
    N, d = size(data)
    N < dims && throw(ArgumentError("PCA dimensions must be less than $N, chosen: $dims")) 
    println("Reducing with PCA to $dims dims")
    data_std = data - repeat(mean(data,dims=1),N,1)
    e, v = eigen(transpose(data_std) * data_std);
    v = v[:,sortperm(e,rev=true)]
    Y = data_std * v[:,1:dims]
end

function H_sigma(D::Vector{Float64},sigma::Float64)
    P = exp.(-D * sigma) 
    sum_p = sum(P)
    H = log(sum_p) + sigma * sum(D .* P) / sum_p
    P = P / sum_p
    return H, P
end

function binary_search(data::Matrix{Float64},perp::Float64,tol::Float64,max_iter::Int;v::Bool)
    N, d = size(data)
    sigma = ones(N)
    P = zeros(N,N)
    sum_x = sum(data.^2,dims=2)
    D = transpose(-2. * data * transpose(data) .+ sum_x) .+ sum_x
    log_perp = log(perp)
    for i in 1:N
        if v && i % 10 == 0
            println("Perplexity search $i / $N")
        end
        sigma_low = -Inf
        sigma_high = Inf
        idx = vcat(1:i-1,i+1:N)
        # println(idx)
        D_i = D[i,idx]
        H, probs = H_sigma(D_i,sigma[i])
        delta_H = H - log_perp
        iter = 0
        while abs(delta_H) > tol && iter < max_iter
            # Perform binary update
            if delta_H > 0.
                sigma_low = sigma[i]
                (sigma_high == Inf || sigma_high == -Inf) && (sigma[i] = sigma[i] * 2.)
                (sigma_high != Inf && sigma_high != -Inf) && (sigma[i] = (sigma[i] + sigma_high) / 2.)
            else
                sigma_high = sigma[i]
                (sigma_low == Inf || sigma_low == -Inf) && (sigma[i] = sigma[i] / 2.)
                (sigma_low != Inf && sigma_low != -Inf) && (sigma[i] = (sigma[i] + sigma_low) / 2.)
            end

            H, probs = H_sigma(D_i, sigma[i])
            delta_H = H - log_perp
            iter += 1
        end
        P[i,idx] = probs
    end
    P, sigma
end

function shannon_entropy(p_distr::Matrix{Float64})
    sh = p_distr.* log2.(p_distr)
    for i in 1:size(sh)[1]
        sh[i,i] = 0.
    end
    -sum(sh,dims=2)
end

perplexity(p_distr::Matrix{Float64}) = 2 .^shannon_entropy(p_distr)

function cost_KL(p_distr::Matrix{Float64},q_distr::Matrix{Float64})
    l = log.(p_distr./q_distr)
    for i in 1:size(l)[1]
        l[i,i] = 0.
    end
    cost = sum(p_distr.*l)
end

function cond_t(data::Matrix{Float64};df::Int64=1)
    sum_x = sum(data.^2,dims=2)
    D = transpose(-2. * data * transpose(data) .+ sum_x) .+ sum_x
    num = 1. ./ (1. .+ D)
    for i in 1:size(num)[1]
        num[i,i] = 0.
    end
    max.(num/sum(num),1e-12), num
end

function cond_gauss(data::Matrix{Float64};sigma::Float64=1.0)
    exp_mat = exp.(-eu_dist(data)/(2*sigma^2))
    for i in 1:size(exp_mat)[1]
        exp_mat[i,i] = 0.
    end
    prob = max.(exp_mat/sum(exp_mat),1e-12)
    prob, exp_mat
end

function eu_dist(data::Matrix{Float64})
    dist_mat2 = zeros(size(data)[1],size(data)[1])
    for i in 1:size(data)[1], j in i:size(data)[1]
        dist_mat2[i,j] = sum((data[i,:] - data[j,:]).^2)
        dist_mat2[j,i] = dist_mat2[i,j]
    end
    dist_mat2
end

function random_start(sample_size::Int,dim::Int;normal::Bool=false,seed::Bool=false)
    if seed
        quote
            import Random
        end
        Random.seed!(1234)
    end
    normal ? randn(sample_size,dim) : rand(sample_size,dim)
end

end # module
