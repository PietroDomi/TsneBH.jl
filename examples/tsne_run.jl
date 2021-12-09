using TsneBH

N = 200
X_dim = 100
Y_dim = 2

X = randn(N,X_dim);

Y = tsne(X,Y_dim,100;lr=1.)
