import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# data
data = [[1.6, 51, 'Small'], [1.7, 62, 'Large'], [1.85, 69, 'Large'], [1.42, 64, 'Small'], [1.3, 65, 'Large'], [2.1, 56, 'Large'], [1.4, 58, 'Small'], [1.65, 57, 'Large'], [1.9, 55, 'Large']]
df = pd.DataFrame(data, columns = ['Height', 'Weight', 'T-Shirt size'])

# features and target
X = df[['Height', 'Weight']]
y = df['T-Shirt size']

# fit model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X, y)

# prediction
test_data = [[1.5, 60]]
test_data = pd.DataFrame(test_data, columns = ['Height', 'Weight'])
prediction = knn.predict(test_data)
print("The T-Shirt size of the customer is: ", prediction[0])

# distances
distances, indices = knn.kneighbors(test_data)
print("The distances between the customer data and other data points are: ", distances[0])

