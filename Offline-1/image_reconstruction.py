import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def low_rank_approximation(A, k):
    U,D,V = np.linalg.svd(A)
    # print(D)
    A_k = np.dot(U[:,:k],np.dot(np.diag(D[:k]),V[:k,:]))
    return A_k

m=0
n=0
img = cv.imread('image.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('image', gray)
cv.waitKey(0)
cv.destroyAllWindows()
print(gray)
U,D,V = np.linalg.svd(gray,full_matrices=False)
print(D)
m=len(gray)
n=len(gray[0])
min_mn=min(m,n)
print(min_mn)
print (U.shape, D.shape, V.shape)
smat=np.diag(D)
#reconstruct A from U,D,V
A_reconstructed=np.dot(np.dot(U,smat),V)
print("A reconstructed: ",A_reconstructed)

#check if A=A_reconstructed
print("A==A_reconstructed: ",np.allclose(gray,A_reconstructed))
# cv.imshow('image', A_reconstructed)
# cv.waitKey(0)
# cv.destroyAllWindows()
j=1
# plt.figure(dpi=200)
for i in range(1,95,5):
    k=i
    print(k)
    gray_k = low_rank_approximation(gray, k)
    plt.subplot(5,4,j)
    j=j+1
    plt.imshow(gray_k, cmap='gray')
    plt.title('k = %s' % k)
    # plt.show()
    # cv.imshow('image', gray_k)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
grey_k = low_rank_approximation(gray, min_mn)

print(min_mn)
plt.subplot(5,4,j)
plt.imshow(grey_k, cmap='gray')
plt.title('k = %s' % min_mn)
# plt.savefig('image_reconstruction.png')
plt.show()
# gray_k = low_rank_approximation(gray, k)
# cv.imshow('image', gray_k)
# cv.waitKey(0)

