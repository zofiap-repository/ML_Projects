from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits

digits = load_digits()
data = scale(digits.data) # normalized data

model = KMeans(n_clusters=10, init='random', n_init=10)
model.fit(data)

# We could add another scanned picture, and 
# see which cluster this picture belongs to
# model.predict([...])