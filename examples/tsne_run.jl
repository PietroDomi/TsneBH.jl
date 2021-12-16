using TsneBH

N = 500
X_dim = 100
Y_dim = 2

# X = randn(N,X_dim);
X = TsneBH.random_start(N,X_dim,normal=true,seed=true)

Y = tsne(X,Y_dim,1000;lr=0.1,verbose=true,exag_fact=1.5)
