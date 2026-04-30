# 🏡 California House Price Predictor

A machine learning web application that predicts California housing prices using a Random Forest model trained on the [California Housing Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices). Built with scikit-learn and served via a Flask web interface.

---

## 📁 Project Structure

```
├── housing.csv              # California Housing Dataset
├── train_model.py           # Pipeline definition + model training + saving
├── preprocessing.py         # Preprocessing pipeline components
├── main.py                  # CLI interface for predictions
├── app.py                   # Flask web app
├── HouseRatePrediction.ipynb # Exploratory analysis & model comparison
├── prod_random_forest.pkl   # Saved production model (generated after training)
└── requirement.txt          # Python dependencies
```

---

## 🧠 Model

| Model               | R² Score |
|---------------------|----------|
| Linear Regression   | 0.636    |
| **Random Forest**   | **0.818** |
| Gradient Boosting   | 0.775    |
| SVR                 | -0.065   |
| KNN                 | 0.266    |

Random Forest was selected as the production model.

**Pipeline steps:**
1. **Geospatial clustering** — custom `Similarity4Cluster` transformer using KMeans + RBF kernel on lat/lon
2. **Categorical encoding** — OneHotEncoder for `ocean_proximity`
3. **Numerical preprocessing** — Median imputation + StandardScaler
4. **RandomForestRegressor** — 150 estimators, max depth 12

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

### 3. Train the model

This generates `prod_random_forest.pkl` required by both the CLI and web app.

```bash
python train_model.py
```

---

## 🚀 Usage

### Web App (Flask)

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser. Fill in the form with housing details to get a price estimate.

### CLI

```bash
python main.py
```

You'll be prompted to enter housing details interactively.

---

## 🔢 Input Features

| Feature              | Description                                      | Example       |
|----------------------|--------------------------------------------------|---------------|
| `longitude`          | Longitude of the block                           | `-122.23`     |
| `latitude`           | Latitude of the block                            | `37.88`       |
| `housing_median_age` | Median age of houses in the block                | `41.0`        |
| `total_rooms`        | Total number of rooms in the block               | `880.0`       |
| `total_bedrooms`     | Total number of bedrooms in the block            | `129.0`       |
| `population`         | Population of the block                          | `322.0`       |
| `households`         | Number of households in the block                | `126.0`       |
| `median_income`      | Median income (in tens of thousands USD)         | `8.33`        |
| `ocean_proximity`    | Proximity to ocean                               | `NEAR BAY`    |

**Valid values for `ocean_proximity`:** `NEAR BAY`, `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `ISLAND`

---

## 📊 Dataset

- **Source:** California Housing Dataset (1990 Census)
- **Rows:** 20,640
- **Target:** `median_house_value` (USD)
- **Missing values:** `total_bedrooms` — imputed with median

---

## 📦 Key Dependencies

- `scikit-learn` — ML pipeline and model
- `pandas` / `numpy` — Data handling
- `Flask` — Web server
- `joblib` — Model serialization

---

## ⚠️ Known Limitations

- The dataset is from the **1990 California Census** — predictions reflect 1990 market conditions and should not be used for current real estate estimates.
- The model was not trained on `total_rooms`, `total_bedrooms`, `population`, or `households` directly in the production pipeline — only `median_income`, `housing_median_age`, and `population` pass through the numerical pipeline alongside geo and categorical features.
- No input validation on the web form — non-numeric inputs will cause a server error.

---

## 📄 License

MIT License. See `LICENSE` for details.
