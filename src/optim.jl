
function gradient_descent(T::Int64, P::Matrix{Float64}, data_red::Matrix{Float64},
                            lr::Float64, exag_fact::Float64;
                            v::Bool, tol::Float64, momentum::Float64,
                            use_trees::Bool, X::Matrix{Float64}, theta::Float64, 
                            perp::Float64, sigma::Vector{Float64})
    # Parameters
    dr = copy(data_red)
    dy_m = zeros(size(dr))
    t = 0
    delta_C = Inf
    dy = Inf
    if use_trees
        P, neighbors = cond_gauss_sparseVP(X, sigma, perp)
        P_joint = joint_gauss_sparse(P)
        P = Array(P)
    end
    # Early exaggeration
    P = P * exag_fact
    q_distr, num = cond_t(dr)
    # Initial cost
    C = cost_KL(P, q_distr)
    println("Beginning gradient descent...")
    while t < T && delta_C > tol && maximum(dy) > 1e-2
        size(P) != size(q_distr) && throw(ArgumentError("sizes don't match"))
        # Compute gradient
        global t = 0 # create recursion check
        dy = use_trees ? gradient_trees(dr, P_joint, neighbors, theta=theta) : grad_KL(P,dr,q_distr,num)
        global t = 0 # reset recursion check
        # Update values
        dy_m = dy_m * momentum - lr * dy
        dr = dr - dy_m
        # Center in zero
        dr = dr - repeat(sum(dr, dims=1) / size(dr)[1], size(dr)[1], 1)
        # Compute loss
        q_distr, num = cond_t(dr)
        C_t = cost_KL(P, q_distr)
        # update iteration
        t += 1
        delta_C = abs(C - C_t)
        C = C_t
        if v && t % 10 == 0
            println("t = $t, C = $C_t, ΔC = $delta_C, min(dy) = $(minimum(dy)), max(dy) = $(maximum(dy))")
        end
        if t == 200
            # Stop exaggeration
            P = P / exag_fact
            println("Stop exaggerating...")
        end
    end
    dr
end

function grad_KL(P::Matrix{Float64}, data_red::Matrix{Float64}, q_distr::Matrix{Float64}, num::Matrix{Float64})
    PQ = P - q_distr
    dy = zeros(size(data_red))
    for i in 1:size(dy)[1]
        dy[i,:] = sum(repeat(PQ[:,i] .* num[:,i], 1, size(data_red)[2]) .* (transpose(data_red[i,:]) .- data_red), dims=1)
    end
    dy
end

function PCA(data::Matrix{Float64}, dims::Int)
    N, d = size(data)
    N < dims && throw(ArgumentError("PCA dimensions must be less than $d, chosen: $dims")) 
    println("Reducing with PCA to $dims dims")
    data_std = data - repeat(mean(data, dims=1), N, 1)
    e, v = eigen(transpose(data_std) * data_std);
    v = v[:, sortperm(e, rev=true)]
    Y = data_std * v[:,1:dims]
end

function H_sigma(D::Vector{Float64}, sigma::Float64)
    P = exp.(-D * sigma) 
    sum_p = sum(P)
    H = log(sum_p) + sigma * sum(D .* P) / sum_p
    P = P / sum_p
    return H, P
end

function binary_search(data::Matrix{Float64}, perp::Float64, tol::Float64, max_iter::Int;v::Bool)
    N, d = size(data)
    sigma = ones(N)
    P = zeros(N, N)
    sum_x = sum(data.^2, dims=2)
    D = transpose(-2. * data * transpose(data) .+ sum_x) .+ sum_x
    log_perp = log(perp)
    for i in 1:N
        if v && i % 100 == 0
            println("Perplexity search $i / $N")
        end
        sigma_low = -Inf
        sigma_high = Inf
        idx = vcat(1:i-1, i+1:N)
        D_i = D[i, idx]
        H, probs = H_sigma(D_i, sigma[i])
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
