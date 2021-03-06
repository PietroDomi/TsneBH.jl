using TsneBH

N = 500
X_dim = 100
Y_dim = 2

# X = randn(N,X_dim);
X = random_start(N, X_dim, normal=true, seed=true)

Y = tsne(X, Y_dim, 500;
         lr=1e-5, tol=5e-4, verbose=true, pca_dim=X_dim÷2,
         exag_fact=1.2, momentum=1e-5, use_trees=false)
