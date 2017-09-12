library(ggplot2)

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
# X = np.zeros((N*K,D)) # data matrix (each row = single example)
X = matrix(0, nrow=N*K, ncol=D)
# y = np.zeros(N*K, dtype='uint8') # class labels
Y = matrix(0, nrow=N*K, ncol=1)
# for j in xrange(K):
train = data.frame(X=c(),Y=c(), K=c())
for (j in 1:K) {
  # ix = range(N*j,N*(j+1))
  ix = (N*(j-1)+1):((N*j))
  # r = np.linspace(0.0,1,N) # radius
  r = seq(0, 1, length=N)
  # t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  t = seq((j-1)*4, j*4, length=N) + (0.2*rnorm(N))
  # X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  # y[ix] = j
  X[ix,1] = r*sin(t)
  X[ix,2] = r*cos(t)
  Y[ix] = j
# lets visualize the data:
}
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
ggplot(data.frame(X=X[,1], Y=X[,2], K=as.factor(Y)), aes(x=X, y=Y, color=K)) +
  geom_point()

