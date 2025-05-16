# Import all libraries like pandas, numpy, matplotlib, seaborn and sci-kit learn.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the housing dataset.
data_path = "Housing_Dataset_Regression.csv"
df = pd.read_csv(data_path)

# Clean the column names.
df.columns = df.columns.str.strip()

# Drop any unnecessary columns present.
if 'No.' in df.columns:
    df.drop(columns=['No.'], inplace=True)

# Set the target column.
target = "median_house_value"
if target not in df.columns:
    raise KeyError(f"Target column '{target}' not found in dataset. Available columns: {df.columns}")

# Identify the numerical and categorical features.
num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_features.remove(target)  # Remove target from numeric features list
cat_features = df.select_dtypes(include=['object']).columns.tolist()

# Numeric features pipeline - impute missing values and scale.
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features pipeline -impute missing values and one-hot encode.
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combined column transformer.
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

X = df.drop(columns=[target])  # Features
y = df[target]  # Target

# Split data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=190/len(df), shuffle=False)

# Dictionary of models to train.
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}

# Hyperparameter grids for tuning.
param_grid = {
    "Ridge Regression": {
        "regressor__alpha": [0.1, 1.0, 10.0]
    },
    "Random Forest Regressor": {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5]
    }
}

# Training models and evaluating them:

# Store the results and predictions.
results = {}
all_preds = {}
rf_feature_names, rf_importances = None, None

for name, model in models.items():
    # Create pipeline with preprocessing and model.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Hyperparameter tuning with GridSearchCV.
    if name in param_grid:
        print(f"\nTuning hyperparameters for {name}")
        grid_search = GridSearchCV(
            pipeline,
            param_grid[name],
            cv=3,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best hyperparameters for {name}:")
        print(grid_search.best_params_)
    else:
        best_model = pipeline.fit(X_train, y_train)

    # The predictions and evaluation.
    y_pred = best_model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    }

    print(f"\n{name} Performance:")
    print(pd.DataFrame([results[name]]))
    print("-" * 60)

    # Save the predictions for the scatter graph.
    all_preds[name] = (y_test, y_pred)

    # Save feature importances for Random Forest Regressor.
    if name == "Random Forest Regressor":
        rf_model = best_model.named_steps["regressor"]
        rf_feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
        rf_importances = rf_model.feature_importances_

# Create a DataFrame for model evaluation results in table format.
df_results = pd.DataFrame(results).T
print("\nOverall Model Comparison:")
print(df_results)

# Graph - Predicted vs actual graph for all models.
plt.figure(figsize=(7, 6))
for model_name, (y_true, y_pred) in all_preds.items():
    sns.scatterplot(x=y_true, y=y_pred, label=model_name, alpha=0.5)

# Implement diagonal reference line of best fit.
min_val = min(y_test.min(), min(pred[1].min() for pred in all_preds.values()))
max_val = max(y_test.max(), max(pred[1].max() for pred in all_preds.values()))
plt.plot([min_val, max_val], [min_val, max_val], '--', color='black')

plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Predicted vs Actual - All Models")
plt.legend()
plt.tight_layout()
plt.show()