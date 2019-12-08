# Gene expression dataset
X= as.matrix(mydatGE)
X= as.matrix(inputge)
x1 = do.lscore(X,k)

# Methylation dataset

X=as.matrix( mydatME)
X= as.matrix(inputcnv)
x2 = do.lscore(X, k)

# miRNA dataset
X= as.matrix(mydatMI)
X= as.matrix(inputmiRNA)
x3 = do.lscore(X, k)
