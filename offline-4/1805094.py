import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
#import multivariate_normal
import scipy.stats as stats
import matplotlib.animation as animation


#take filename as input

file=input("Enter the filename: ")
if file=='':
    file='dataset/6D_data_points.txt'
#open the file
f=open(file,"r")
#read comma separated values in each line into np array
data=np.array([line.strip().split(',') for line in f.readlines()],dtype=float)
#print first 5 rows
N=data.shape[0]
M=data.shape[1]
print("Number of data points: ",N)
print("Number of dimensions: ",M)

#dont perform PCA if M is 2 or less, just plot the data
if(M<=2):
    data_proj=data
    plt.scatter(data[:,0],data[:,1])
    
    # Set the frequency of the x and y axis labels
    x_step = (max(data[:,0]) - min(data[:,0])) / 10  
    y_step = (max(data[:,1]) - min(data[:,1])) / 10  

    plt.xticks(np.arange(min(data[:,0]), max(data[:,0]), step=x_step))
    plt.yticks(np.arange(min(data[:,1]), max(data[:,1]), step=y_step))

    plt.savefig(file.split('.')[0]+'.png')

else:
    normalized_data=data-np.mean(data,axis=0)
    # Perform SVD on the centered data
    U, S, Vt = np.linalg.svd(normalized_data)

    # Project the centered data onto the first two principal components
    data_proj = np.dot(normalized_data, Vt.T[:,:2])

    # Perform PCA using the library function
    pca = PCA(n_components=2)
    data_proj_lib = pca.fit_transform(data)

    # max_abs_cols = np.argmax(np.abs(data_proj), axis=0)
    # signs = np.sign(data_proj[max_abs_cols, range(data_proj.shape[1])])
    # data_proj *= signs

    # Now, data_proj should match data_proj_lib
    print(np.allclose(data_proj, data_proj_lib))

    plt.clf()
    plt.scatter(data_proj[:,0],data_proj[:,1])

    # x_step = (max(data_proj[:,0]) - min(data_proj[:,0])) / 10 
    # y_step = (max(data_proj[:,1]) - min(data_proj[:,1])) / 10

    # plt.xticks(np.arange(min(data_proj[:,0]), max(data_proj[:,0]), step=x_step))
    # plt.yticks(np.arange(min(data_proj[:,1]), max(data_proj[:,1]), step=y_step))

    plt.savefig(file.split('.')[0]+'_pca.png')

#apply GMM on the projected data
#take number of clusters as input


K=int(input("Enter the number of clusters: "))

flag_animate=input("Do you want to animate the clustering process? (y/n): ")
if flag_animate.lower()=='y':
    flag_animate=True
else:
    flag_animate=False

#if k is none, use lower limit as 3 and upper limit as 8
if K==0:
    lower_limit=3
    upper_limit=9
else:
    lower_limit=K
    upper_limit=K+1

# best_K=int(input("Enter the best number of clusters: "))
# Define the Gaussian PDF
def gaussian_pdf(x, mean, cov):
    return np.exp(-0.5 * np.dot(np.dot(x - mean, np.linalg.inv(cov)), x - mean).T) / np.sqrt(np.linalg.det(cov))

# Number of dimensions
dim = 2

#define seed for reproducibility
np.random.seed(94)




# Best log-likelihood
best_log_likelihood = -np.inf
best_clusters = None
likelihoods=[]
clusters=[]
for K in range(lower_limit,upper_limit):
    # Run the EM algorithm 5 times
    for t in range(5):
        # Initialize the parameters
        mix = np.ones(K) / K
        mean = np.random.rand(K, dim)
        cov = np.array([np.eye(dim) for _ in range(K)])
        prev_log_likelihood = -np.inf
        probs=np.zeros((len(data_proj), K))
        # if(_==0):
        #     print('Initial mean: ',mean)
        #     print('Initial covariance: ',cov)
        #     print('Initial mixing coefficients: ',mix)
        f=False
        # output_file='log_likelihood_'+str(_)+'.txt'
        # f=open(output_file,'w')
        # EM algorithm
        itr=0
        resp_list=[]

        while True :
            # E-step
            resp = np.zeros((len(data_proj), K))
            # if f==False and _==0:
            #     print('resp: ',resp)
            #     f=True
            for j in range(K):
                reg_cov = cov[j] + np.eye(cov[j].shape[0]) * 1e-6
                multivariate_normal = stats.multivariate_normal(mean[j], reg_cov).pdf(data_proj)
                # resp[i, j] = mix[j] * gaussian_pdf(data_proj[i], mean[j], cov[j])
                resp[:, j] = mix[j] * multivariate_normal
            column_sums = resp.sum(axis=1)
            resp /= resp.sum(axis=1, keepdims=True)
            resp_list.append(resp)
            # M-step
            for j in range(K):
                # Update the mixing coefficients
                mix[j] = resp[:, j].mean()
                # Update the means
                mean[j] = np.dot(resp[:, j], data_proj) / resp[:, j].sum()
                # Update the covariance matrices
                cov[j] = np.dot(resp[:, j] * (data_proj - mean[j]).T, data_proj - mean[j]) / resp[:, j].sum()
            # Compute the log-likelihood
            log_likelihood=np.sum(np.log(column_sums))
            # f.write(str(log_likelihood)+', '+str(prev_log_likelihood)+'\n')
            # f.flush()
            if abs(log_likelihood - prev_log_likelihood) < 1e-2:
                print('value of log likelihood: ',log_likelihood)
                break
            prev_log_likelihood = log_likelihood
            # itr+=1
            # if itr==100:
            #     print('value of log likelihood: ',log_likelihood)
            #     break

            # cls=np.argmax(resp,axis=1)
        if flag_animate:
            fig, ax = plt.subplots()

            def animate(i):
                ax.clear()
                cls = np.argmax(resp_list[i], axis=1)
                scatter = ax.scatter(data_proj[:, 0], data_proj[:, 1], c=cls, cmap='viridis')
                ax.set_title('Iteration: '+str(t)+'K: '+str(K))
                return scatter,
            ani = animation.FuncAnimation(fig, animate, frames=len(resp_list), interval=200, blit=True,repeat=False)

            plt.show()
        # If this log-likelihood is the best, save the clusters
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_clusters = resp.argmax(axis=1)
    likelihoods.append(best_log_likelihood)
    clusters.append(best_clusters)

if lower_limit==upper_limit-1:
    print('value of best log likelihood: ',best_log_likelihood)
    print('clusters: ',best_clusters)
    plt.clf()
    # Plot the best clusters
    plt.scatter(data_proj[:, 0], data_proj[:, 1], c=clusters[0],cmap='viridis')
    plt.colorbar()
    plt.savefig(file.split('.')[0]+'_gmm.png')
    exit(0)
# Plot the log-likelihoods
plt.clf()
plt.plot(range(3,9),likelihoods)
plt.xlabel('Number of clusters')
plt.ylabel('Log-likelihood')
plt.savefig(file.split('.')[0]+'_log_likelihood.png')

# plt.clf()
# # Plot the best clusters
# plt.scatter(data_proj[:, 0], data_proj[:, 1], c=clusters[best_K-3],cmap='viridis')
# plt.colorbar()


# plt.savefig(file.split('.')[0]+'_gmm.png')

for i in range(3,9):
    plt.clf()
    # Plot the best clusters
    plt.scatter(data_proj[:, 0], data_proj[:, 1], c=clusters[i-3],cmap='viridis')
    plt.colorbar()
    plt.savefig(file.split('.')[0]+'_gmm_'+str(i)+'.png')


