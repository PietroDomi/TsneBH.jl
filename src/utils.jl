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