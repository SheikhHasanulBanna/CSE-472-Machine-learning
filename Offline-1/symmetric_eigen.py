import numpy as np


#take n as input
n=int(input("Enter the value of n: "))
print(n)
#produce random n*n matrix where each entry is integer
A=np.random.randint(10,size=(n,n))
#make A symmetric
A=A+A.T
# make A diagonally dominant
for i in range(n):
    A[i][i]=np.sum(np.absolute(A[i]))-np.absolute(A[i][i])+1

#check if A is invertible
print("matrix A: ",A)


eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A are: ",eigenvalues)
print("Eigenvectors of A are: ",eigenvectors)
#form diagonal matrix with eigenvalues
D=np.diag(eigenvalues)
#reconstruct A
A_reconstructed=np.dot(np.dot(eigenvectors,D),eigenvectors.T)
print("A reconstructed: ",A_reconstructed)
print("A: ",A)
#check if A=A_reconstructed
print("A==A_reconstructed: ",np.allclose(A,A_reconstructed))
