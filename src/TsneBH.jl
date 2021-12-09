module TsneBH

export tsne

using LinearAlgebra

function tsne(X::Matrix{Float64},emb_size::Int64,T::Int64;lr::Float64=1.)
    # Create an initial embedding
    Y = randn(size(X)[1],emb_size)
    # Start the iteration
    gradient_descent(T,X,Y,lr)
end

function gradient_descent(T::Int64,data::Matrix{Float64},data_red::Matrix{Float64},lr::Float64)
    # Start iterating
    dr = copy(data_red)
    for t in 1:T
        # Compute gradient
        dy = grad_KL(data,dr)
        # Update values
        dr = dr - lr * dy
        # Compute loss
        if t % 10 == 0
            println("Iteration $t, cost=$(cost_KL(cond_gauss(data)[1],cond_t(dr)[1]))")
        end
        # Center in zero
        dr = dr - repeat(sum(dr,dims=1)/size(dr)[1],size(dr)[1],1)
    end
    dr
end

function grad_KL(data::Matrix{Float64},data_red::Matrix{Float64})
    q_distr, num = cond_t(data_red)
    PQ = cond_gauss(data)[1] - q_distr
    dy = zeros(size(data_red))
    for i in 1:size(dy)[1]
        dy[i,:] = sum(repeat(PQ[:,i] .* num[:,i],1,size(data_red)[2]) .* (transpose(data_red[i,:]) .- data_red),dims=1)
    end
    dy
end

function shannon_entropy(p_distr::Matrix{Float64})
    sh = p_distr.* log2.(p_distr)
    for i in 1:size(sh)[1]
        sh[i,i] = 0.
    end
    -sum(sh,dims=2)
end

perplexity(p_distr::Matrix{Float64}) = 2 .^shannon_entropy(p_distr)

cost_KL(p_distr::Matrix{Float64},q_distr::Matrix{Float64}) = sum(p_distr.*(log.(p_distr./q_distr)))

function cond_t(data::Matrix{Float64};df::Int64=1)
    sum_x = sum(data.^2,dims=2)
    num = -2. * data * transpose(data)
    num = 1. ./ (1. .+ (transpose(num .+ sum_x) .+ sum_x))
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

function eu_dist(data::Array{Float64,2})
    dist_mat2 = zeros(size(data)[1],size(data)[1])
    for i in 1:size(data)[1], j in i:size(data)[1]
        dist_mat2[i,j] = sum((data[i,:] - data[j,:]).^2)
        dist_mat2[j,i] = dist_mat2[i,j]
    end
    dist_mat2
end

end # module
