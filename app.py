from flask import Flask , render_template, request
# import main
# from main import final_model_reloaded as model

import pandas as pd
import numpy as np
import joblib
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

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

    

model = joblib.load("prod_random_forest.pkl")



app = Flask(__name__)

@app.route("/" , methods= ["GET" , "POST"])
def Home():
    if request.method == "POST":
      longitude =  request.form.get("Longitude")
      latitude =  request.form.get("Latitude")
      housing_median_age =  request.form.get("Housing_Median_Age")
      total_rooms =  request.form.get("Total_Rooms")
      total_bedrooms =  request.form.get("Total_Bedrooms")
      population =  request.form.get("Population")
      households =  request.form.get("Households")
      median_Income =  request.form.get("Median_Income")
      ocean_proximity =  request.form.get("Ocean_Proximity")


      # prediction logic

      data = np.array([longitude, latitude, housing_median_age, total_rooms,
          total_bedrooms, population, households,median_Income,
          ocean_proximity ]).reshape(1,-1)

      clms = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
             'total_bedrooms', 'population', 'households', 'median_income',
             'ocean_proximity']
      df = pd.DataFrame(data , columns = clms)


    #   val = model.predict(df)
    #   val = f" Your House Price is Around {val[0]}"

      prediction = model.predict(df)[0]
      formatted_price = f"${prediction:,.2f}"
      return render_template('index.html', 
                                 House_Value=f"Estimated House Price: {formatted_price}")
        

     


    #   return render_template('index.html', House_Value = val)
     
       
    return render_template("index.html")
if __name__ == "__main__":
  app.run(debug=True)



