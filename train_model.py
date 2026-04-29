import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

# ===========================================
# Define Custom Transformer (Same as in Deployment)
# ===========================================
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

# ===========================================
# Preprocessing Pipeline
# ===========================================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessing = ColumnTransformer([
    ("geo", Similarity4Cluster(n_clusters=8), ["latitude", "longitude"]),
    ("cat", cat_pipeline, ["ocean_proximity"]),
    ("num", num_pipeline, ["median_income", "housing_median_age", "population"])
])

# ===========================================
# Model Training & Saving
# ===========================================
# Load data
housing = pd.read_csv('housing.csv')
X_train, X_test, y_train, y_test = train_test_split(
    housing.drop("median_house_value", axis=1),
    housing["median_house_value"],
    test_size=0.2,
    stratify=housing["ocean_proximity"],
    random_state=42
)

# Define and train model
model = Pipeline([
    ("preprocessing", preprocessing),
    ("model", RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=8,
        max_features=4,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "prod_random_forest.pkl")

# # Validate
# test_pred = model.predict(X_test)
# print(f"Test R²: {r2_score(y_test, test_pred):.4f}")
# print(f"Test RMSE: {mean_squared_error(y_test, test_pred, squared=False):.1f}")