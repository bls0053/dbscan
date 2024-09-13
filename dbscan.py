
import matplotlib.pyplot as plt
import matplotlib.patches as p
import pandas as pd

import numpy as np
import random
import math
from array import *

# Point coloring
import seaborn as sns

# Generates sample data
from sklearn import datasets



# Main DBSCAN logic
def dbscan(eps, minpts, df):

    # Univisited points
    unvisited = list(df.index)
    # Stack to handle neighbors
    stack = set()
    # Final cluster array
    clusters = [[]]
    # Current cluster count (Noise is of cluster '0')
    curr_c = 1
    # Maintains edge case points (border & first)
    b_list = []

    #Runs until no more points to check
    while (len(unvisited) != 0):

        first = True
        stack.add(random.choice(unvisited))
        clus_elements = []
        
        while (len(stack) != 0):
            i = stack.pop()

            # Temporary list to check previous b.p.
            temp = []
            
            # Returns point type and neighbors
            neighbors, core, border, noise = find_neigh(eps, minpts, df, i)

            # Handles border point edge case *PE
            if border & first:
                unvisited.remove(i)
                b_list.append(i)
                continue

            unvisited.remove(i)

            # Updates neighbors with only unvisited and previous border points *PE
            if (len(b_list) != 0):
                temp = set(neighbors) & set(b_list)
                for j in temp:
                    b_list.remove(j)
                    unvisited.append(j)

            
            neighbors = set(neighbors) & set(unvisited)
            neighbors = set(neighbors) | set(temp)

            # Updates relevant cluster
            if (core):
                first = False
                stack.update(neighbors)
                clus_elements.append(i)
 
            elif (border):
                clus_elements.append(i)
                continue

            elif (noise):
                clusters[0].append(i)
                continue

        # Adds cluster elements to list of completed clusters
        if not first:
            clusters.append([])
            clusters[curr_c].extend(clus_elements)
            curr_c += 1

    # Adds 'unused' border points as noise
    clusters[0].extend(b_list)
    return clusters     


# Finds neighbors and point type
def find_neigh(eps, minpts, df, i):

    # Retrieves point from dataframe
    x1,y1 = df.iloc[i]['X'], df.iloc[i]['Y']
    neigh = []

    # Calculates eucldiean distance to all other points (Could be made faster?)
    for j in range (0,len(df)):
        if (j == i):
            continue
        
        x2,y2 = df.iloc[j]['X'], df.iloc[j]['Y']
        dist = math.sqrt( ((x2-x1)**2) + ((y2-y1)**2) )
         
        if dist <= eps:
            neigh.append(j)
    
    # Return
    if (len(neigh) == 0):
        return (neigh, False, False, True)
    
    elif (len(neigh) < minpts-1 and len(neigh) > 0):
        return (neigh, False, True, False)

    elif (len(neigh) >= minpts-1):
        return (neigh, True, False, False)


# Init variables for DBSCAN/Plotting
    
    # big groups
    
# n_samples = 500
# seed = 10
# eps = .25
# minpts = 5
# data = []

# n_samples = 200
# seed = 10
# eps = .15
# minpts = 20
# data = []
    
    # small groups 
n_samples = 200
seed = 10
eps = .18
minpts = 5
data = []

    # sparse groups
# n_samples = 50
# seed = 10
# eps = .3667
# minpts = 3
# data = []

centers = [(1, 1), (3, 3) , (1,3), (2,1)]
cluster_std = [.45, .1, .3, .5]
X, y= datasets.make_blobs(n_samples, cluster_std=cluster_std, centers=centers, n_features=2, random_state=seed)
data.append(X)

X,y = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
data.append(X)

X,y = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
data.append(X)

rng = np.random.RandomState(seed)
X,y = rng.rand(n_samples, 2), None
data.append(X)

fig, axes = plt.subplots(2, 4, figsize=(14, 8))

titles = ["Blobs", "Noisy Circles", "Noisy Moons",  "Random", 
          "Blobs DBSCAN", "Noisy Circles DBSCAN", "Noisy Moons DBSCAN",  "Random DBSCAN"]

for i, ax in enumerate(axes.ravel()):
    ax.set_title(titles[i])

# Calls DBSCAN and plots data
for i in range (0,len(data)):

    df = pd.DataFrame(data[i], columns = ["X", "Y"] )

    ax = axes[0,i]
    ax.set_aspect('equal')
    ax.scatter(df['X'], df['Y'], label='', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    fig.suptitle("Distribution Samples", fontsize=20)


    clusters_array = dbscan(eps,minpts,df)

    num_colors = 10
    distinct_colors = sns.color_palette("hsv", n_colors=num_colors).as_hex()

    ax = axes[1, i]
    ax.set_aspect('equal')

    for j, indices in enumerate(clusters_array):

        x_points = [df.loc[idx, 'X'] for idx in indices]
        y_points = [df.loc[idx, 'Y'] for idx in indices]
        
        if (j != 0):
            color = random.choice(distinct_colors)
            distinct_colors.remove(color)

            for k in range(0,len(x_points)):
                circle = p.Circle((x_points[k],y_points[k]), eps, edgecolor='r',facecolor='none')
                ax.add_patch(circle)

            ax.scatter(x_points, y_points, label=f'cluster {j}', color=color)
            
        else:
            color = '000000'

            ax.scatter(x_points, y_points, label='Noise', color=color)
        
        ax.legend(loc='lower center', bbox_to_anchor=(0,0), bbox_transform=axes[1, i].transAxes)


plt.show()


