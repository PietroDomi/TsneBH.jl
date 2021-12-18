module TsneBH

export tsne, random_start

using LinearAlgebra
using Statistics
using Random

include("utils.jl");
include("optim.jl");

"""
tsne(X::Matrix{Float64}, emb_size::Int64, T::Int64;
        lr::Float64 = 1., perp::Float64 = 30., tol::Float64 = 1e-5,
        max_iter::Int = 50,  momentum::Float64 = 0.01, 
        pca::Bool = true, pca_dim::Int = 50, exag_fact::Float64 = 4.,
        use_seed::Bool = false, verbose::Bool = true)

    
"""

function tsne(X::Matrix{Float64}, emb_size::Int64, T::Int64;
                lr::Float64 = 1., perp::Float64 = 30., tol::Float64 = 1e-5,
                max_iter::Int = 50,  momentum::Float64 = 0.01, 
                pca::Bool = true, pca_dim::Int = 50, exag_fact::Float64 = 4.,
                use_seed::Bool = false, verbose::Bool = true)
    if use_seed
        Random.seed!(1234)
    end
    # Create an initial random embedding
    Y = randn(size(X)[1],emb_size)
    # Perform PCA if selected
    pca && (X = PCA(X,pca_dim))
    # Search for the sigmas of the Gaussians
    P, sigma = binary_search(X,perp,tol,max_iter,v=verbose)
    # Start the iteration
    Y_new = gradient_descent(T,P,Y,lr,exag_fact,v=true,tol=tol,momentum=momentum)
end

end # module
