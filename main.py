import numpy as np
import pandas as pd 
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

# Define the custom transformer class FIRST
class Similarity4Cluster(BaseEstimator, TransformerMixin):
    """Custom transformer for geospatial similarity clustering"""
    def __init__(self, n_clusters=10, gamma=0.1, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, 
                             random_state=self.random_state)
        self.kmeans_.fit(X)
        return self
        
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"similarity_with_cluster_{i}" for i in range(self.n_clusters)]





final_model_reloaded = joblib.load("prod_random_forest.pkl")



# User Interaction
if __name__ == "__main__":

  print("Provide the following details of your House : \n\n")
  longitude = float(input("longitude:  "))
  latitude = float(input("latitude:   "))
  housing_median_age= float(input("housing_median_age:  "))
  total_rooms= float(input("total_rooms:  "))
  total_bedrooms= float(input("total_bedrooms:  "))
  population= float(input("population:  "))
  households= float(input("households:  "))
  median_income= float(input("median_income:  "))
  ocean_proximity= input("ocean_proximity:  ")


# prediction logic

  data = np.array([longitude, latitude, housing_median_age, total_rooms,
       total_bedrooms, population, households, median_income,
       ocean_proximity ]).reshape(1,-1)

  clms = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'ocean_proximity']
  df = pd.DataFrame(data , columns = clms)

# prediction
  result = final_model_reloaded.predict(df)

# print the prediction

  print(f"\n\t\t The Price of the this House is Approaximately  around : {result}")