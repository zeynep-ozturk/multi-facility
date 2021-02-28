
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')


#reading the text file containing x and y coordinates of 654 customer locations
import pandas as pd

df = pd.read_csv('p654.txt', skiprows=1, skipfooter=22, engine='python', delimiter = r"\s+", names=['x', 'y', 'c'])
df.drop('c', axis=1, inplace=True)
df=df.values #convert dataframe to array for computational efficiency


# ## K-means clustering


def kmeans(k, df, seed):
    #k: number of clusters
    #df: array containing x and y coordinates of the customers
    #seed: random number generator seed for different initializations
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance, distance_matrix
    #itr=0 #iteration number
    fixed_rng = np.random.RandomState(seed=seed) #fix random number seed for reproducibility
    centers_x = fixed_rng.uniform(df[:,0].min(), df[:,0].max(), k) # x-coordinates of the initial centers
    centers_y = fixed_rng.uniform(df[:,1].min(), df[:,1].max(), k) # y-coordinates of the initial centers
    centers = pd.DataFrame({'x':centers_x,'y':centers_y}).values
    old_centers = np.zeros((k, 2))
    clusters = np.zeros(len(df), dtype=int) #array for the clusters of customers
    total_dist = 0 #total distance from facilities to customers
    while True:
        #print('iteration # ', itr)
        dist_to_centers = distance_matrix(df, centers, p=2) #distance to centers
        for i in range(len(df)):
            closest_center_idx = np.argmin(dist_to_centers[i,:]) #index of closest centroid
            clusters[i]=closest_center_idx
        old_centers = centers.copy()
        #compute new centers
        for j in range(k):
            customers = [df[i,:] for i in range(len(df)) if clusters[i] == j]
            if len(customers)!=0:
                centers[j,:]=np.mean(customers, axis=0) #find the center of clusters
        #itr+=1
        #if the new and old centers are identical stop the algorithm and compute the objective function
        if np.array_equal(old_centers,centers):
            for j in range(k):
                customers = [df[i,:] for i in range(len(df)) if clusters[i] == j]
                #total euclidean distance from customers to their assigned facilities
                if len(customers)!=0:
                    total_dist+=distance.cdist(customers, centers[j,:].reshape(1,-1), metric='euclidean').sum()
            break
    return total_dist, centers, clusters


import time
import numpy as np
cpu_time_list = list()
#5 seeds for 5 initializations are randomly generated
seed_list = np.random.RandomState(seed=10).randint(0,1000,5)
opt_dist_list = [551062.8811,551062.8811,551062.8811,551062.8811,551062.8811,209068.7935,209068.7935,209068.7935,209068.7935,209068.7935,147050.7904,147050.7904,147050.7904,147050.7904,147050.7904]
print(seed_list)
k_list = [3,5,8]
kmeans_dist_list = list()
kmeans_locs_list = list()
kmeans_soln_list = list()
for k in k_list:
    for seed in seed_list:
        start = time.time()
        results = kmeans(k, df, seed)
        kmeans_dist_list.append(results[0])
        kmeans_locs_list.append(results[1])
        kmeans_soln_list.append(results[2])
        done = time.time()
        cpu_time_list.append(done - start)
print(kmeans_dist_list)
print(cpu_time_list)
print((np.asarray(kmeans_dist_list)-np.asarray(opt_dist_list))/np.asarray(opt_dist_list)*100)


import matplotlib.pyplot as plt
colors = ['blue', 'green', 'red', 'black', 'orange', 'purple', 'yellow', 'turquoise']
for f, best in zip(f_list, [2,9,13]):
    plt.figure()
    for i in range(f):
        plt.plot(kmeans_locs_list[best][i,0], kmeans_locs_list[best][i,1], 'o', color=colors[i], markersize=12)
        plt.plot(df[np.where(kmeans_soln_list[best]==i),0][0], df[np.where(kmeans_soln_list[best]==i),1][0], '+', alpha=0.5, color=colors[i])
    plt.title('Customer Assigments with K-means for {} facilities'.format(f))


# ## VARIABLE NEIGHBORHOOD SEARCH

def obj_func(f, df, locs, X):
    from scipy.spatial import distance, distance_matrix
    total_dist=0
    for c in range(f):
        customers = [df[ix,:] for ix in range(len(df)) if X[ix] == c]
        #total euclidean distance from customers to their assigned facilities
        if len(customers)!=0:
            total_dist+=distance.cdist(customers, locs[c,:].reshape(1,-1), metric='euclidean').sum()
    return total_dist

def weiszfeld(locs, df, X, eps):
    #f: number of clusters
    #df: array containing x and y coordinates of the customers
    #locs: x and y coordinates of facility locations
    #eps: convergence parameter
    import copy
    import numpy as np
    new_locs = copy.deepcopy(locs)
    f = len(locs)
    for i in range(f):
        itera=0
        deltax,deltay = eps+1,eps+1
        assigned = df[np.where(X==i)[0],:]
        while abs(deltax)>eps and abs(deltay)>eps:
            x = new_locs[i, 0]; y = new_locs[i, 1]
            a = assigned[:,0]; b = assigned[:,1]
            denom = np.sqrt(((x-a)**2)+((y-b)**2)+eps)
            x_new = np.divide(a, denom).sum()/(sum(1/denom))
            y_new = np.divide(b, denom).sum()/(sum(1/denom))
            deltax = x_new-x
            deltay = y_new-y
            new_locs[i, 0] = x_new
            new_locs[i, 1] = y_new
            itera+=1
    return new_locs


def VNS(f, df, seed, n):
    #k: number of clusters
    #df: array containing x and y coordinates of the customers
    #seed: random number generator seed for different initializations
    #n: number of iterations allowed (without improvement)
    from scipy.spatial import distance, distance_matrix
    import numpy as np
    import pandas as pd
    import copy
    import random

    fixed_rng = np.random.RandomState(seed=seed) #fix random number seed for reproducibility
    locs_x = fixed_rng.uniform(df[:,0].min(), df[:,0].max(), f) # x-coordinates of the initial locs
    locs_y = fixed_rng.uniform(df[:,1].min(), df[:,1].max(), f) # y-coordinates of the initial locs
    locs = pd.DataFrame({'x':locs_x,'y':locs_y}).values
    dist_to_locs = distance_matrix(df, locs, p=2) #distance to locs
    X = np.zeros(len(df), dtype=int) #array for the facility assignments of customers

    for i in range(len(df)):
        closest_center_idx = np.argmin(dist_to_locs[i,:]) #index of closest centroid
        X[i]=closest_center_idx

    locs = weiszfeld(locs, df, X, eps)

    not_improved = 0 #number iterations without improvement
    incumbent = obj_func(f, df, locs, X)
    incumbent_X = copy.deepcopy(X)
    while not_improved<n:
        j=0
        while j<len(Nk):
            k=Nk[j]
            # shaking
            selected = random.choices(list(enumerate(X)), k=k) #indices and facilities of the random customers
            X_prime = copy.deepcopy(X)
            for i in range(k):
                idx, fac = selected[i]
                fac_candid = np.arange(0, f) #find candidate facilities
                fac_candid = np.delete(fac_candid, fac) #delete current facility from candidate list
                fac_prime = np.random.choice(fac_candid, size=1, replace=False) #new facility of the selected customer
                X_prime[idx] = fac_prime # X' is generated
            # local search
            local_imp = np.zeros(len(X_prime))
            for cust,fac in enumerate(X_prime):
                local_imp[cust] = np.min(dist_to_locs[cust, :] - dist_to_locs[cust, fac])
            X_doubleprime = copy.deepcopy(X_prime)
            cust_to_change = np.argmin(local_imp)
            fac_new = np.argmin(dist_to_locs[cust_to_change, :] - dist_to_locs[cust_to_change, fac])
            X_doubleprime[cust_to_change]=fac_new
            #if (obj_func(f, df, locs, X_prime)<obj_func(f, df, locs, X_doubleprime)):
            #    print(obj_func(f, df, locs, X_prime),obj_func(f, df, locs, X_doubleprime))
            #move or not
            obj_X_doubleprime = obj_func(f, df, locs, X_doubleprime)
            obj_X = obj_func(f, df, locs, X)
            if obj_X_doubleprime<obj_X:
                improved=1
                X = copy.deepcopy(X_doubleprime)
                if obj_X_doubleprime<incumbent:
                    incumbent=obj_X_doubleprime #best solution so far
                    incumbent_X=copy.deepcopy(X)
                    incumbent_locs = copy.deepcopy(locs)
                #print('improvement occured: ', obj_X_doubleprime, obj_X, 'incumbent: ', incumbent,'Nk: ',Nk[j])
                j=0
                locs = weiszfeld(locs, df, X, eps)
                dist_to_locs = distance_matrix(df, locs, p=2) #distance to locs
            else:
                improved=0
                j+=1
        if improved==0:
            not_improved+=1
        else:
            not_improved=0
    return incumbent, incumbent_X, incumbent_locs


import time
import numpy as np
Nk = [1,2,3] #selecting the set of neighborhood structures
#5 seeds for 5 initializations are randomly generated
seed_list = np.random.RandomState(seed=10).randint(0,1000,5)
#seed_list = [seed_list[0]]
f_list = [3,5,8]
#f_list = [f_list[2]]
cpu_time_list = list()
opt_dist_list = [551062.8811,551062.8811,551062.8811,551062.8811,551062.8811,209068.7935,209068.7935,209068.7935,209068.7935,209068.7935,147050.7904,147050.7904,147050.7904,147050.7904,147050.7904]
print(seed_list)
vns_dist_list = list()
vns_locs_list = list()
vns_soln_list = list()
n=1000
eps = .5
for f in f_list:
    for seed in seed_list:
        start = time.time()
        result = VNS(f, df, seed, n)
        vns_dist_list.append(result[0])
        vns_soln_list.append(result[1])
        vns_locs_list.append(result[2])
        done = time.time()
        cpu_time_list.append(done - start)
print(vns_dist_list)
print(cpu_time_list)
print((np.asarray(vns_dist_list)-np.asarray(opt_dist_list))/np.asarray(opt_dist_list)*100)


df[np.where(vns_soln_list[0]==1)[0],0]==df[np.where(vns_soln_list[0]==1),0][0]
#df[np.where(X==i)[0],:]


import matplotlib.pyplot as plt
colors = ['blue', 'green', 'red', 'black', 'orange', 'purple', 'yellow', 'turquoise']
for f, best in zip(f_list, [2,9,13]):
    plt.figure()
    for i in range(f):
        plt.plot(vns_locs_list[best][i,0], vns_locs_list[best][i,1], 'o', color=colors[i], markersize=12)
        plt.plot(df[np.where(vns_soln_list[best]==i),0][0], df[np.where(vns_soln_list[best]==i),1][0], '+', alpha=0.5, color=colors[i])
    plt.title('Customer Assigments with VNS for {} facilities'.format(f))
