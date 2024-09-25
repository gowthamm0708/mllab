import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

data = list(zip(x, y))
inertias = []

for i in range(1, len(data) + 1): 
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, len(data) + 1), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
