import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

# Load data
file_path = "1193_P_Trained_models_psnr_result.csv"  # Replace with the correct path
data = pd.read_csv(file_path, header=None)
X = np.array([eval(row[0]) for row in data.values])
y = data.iloc[:, -1]

# Define models and parameters
models = [
    ('Linear Regression', LinearRegression(), {
        'fit_intercept': [True, False]
    }),

    ('Ridge Regression', Ridge(), {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr']
    }),

    ('Lasso Regression', Lasso(), {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'selection': ['cyclic', 'random'],
        'max_iter': [5000, 10000]
    }),

    ('ElasticNet Regression', ElasticNet(), {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.5, 0.8, 1.0],
        'max_iter': [5000, 10000]
    }),

    ('Decision Tree Regression', DecisionTreeRegressor(), {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    }),

    ('Random Forest Regression', RandomForestRegressor(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error']
    }),

    ('XGBoost Regression', GradientBoostingRegressor(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.001, 0.01, 0.1],
        'subsample': [0.7, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2'],
        'loss': ['squared_error', 'huber'],
        'n_iter_no_change': [5, 10]  # 🔹 Early Stopping
    }),

    ('AdaBoost Regression', AdaBoostRegressor(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'loss': ['linear', 'square', 'exponential']  # 🔹 Ajuste de loss
    }),

    ('Extra Trees Regression', ExtraTreesRegressor(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error']
    }),

    ('KNN Regression', KNeighborsRegressor(), {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [20, 30],
        'metric': ['minkowski', 'euclidean', 'manhattan']
    }),

    ('Support Vector Regression', SVR(), {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['linear', 'rbf', 'poly'],
        'epsilon': [0.001, 0.01, 0.1],
        'degree': [2, 3]  # 🔹 Evitar sobreajuste en kernel polinómico
    })
]


# Prepare storage for results
results = []

# Define K-Fold
kfolds = 10
kf = KFold(n_splits=kfolds)

# Model evaluation
for model_name, model, param_grid in models:
    print(f"Processing {model_name}...")
    start_time = time.time()

    # Hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_model_instance = grid_search.best_estimator_

    # Evaluation with MSE and R^2
    cv_scores_mse = cross_val_score(best_model_instance, X, y, cv=kf, scoring='neg_mean_squared_error')
    avg_cv_score_mse = np.abs(np.mean(cv_scores_mse))

    cv_scores_r2 = cross_val_score(best_model_instance, X, y, cv=kf, scoring='r2')
    avg_cv_score_r2 = np.abs(np.mean(cv_scores_r2))

    # Evaluation with C-Index
    c_index_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_model_instance.fit(X_train, y_train)
        y_pred = best_model_instance.predict(X_test)

        c_index_scores.append(concordance_index(y_test, y_pred))

    avg_c_index = np.mean(c_index_scores)

    # Execution time
    duration = time.time() - start_time

    # Store results
    results.append((model_name, duration, avg_cv_score_mse, avg_cv_score_r2, avg_c_index))

    # Print partial results
    print(f"{model_name} - Time: {duration:.2f} sec, MSE: {avg_cv_score_mse:.4f}, R^2: {avg_cv_score_r2:.4f}, C-Index: {avg_c_index:.4f}")

# Plot results
labels, times, mses, r2s, c_indexes = zip(*results)

plt.figure(figsize=(20, 6))

# Training time
#plt.subplot(1, 4, 1)
#plt.barh(labels, times, color='salmon')
#plt.xlabel('Time (seconds)')
#plt.title('Training Time Comparison')

# Mean Squared Error (MSE)
plt.subplot(1, 2, 1)
plt.barh(labels, mses, color='lightgreen')
plt.xlabel('Mean Squared Error')
plt.title('Model Performance (MSE)')

# Set x-axis limits (starting from 15)
plt.xlim(10, max(mses) + 1)  # Adjust the upper bound slightly for spacing

# R^2 Score
#plt.subplot(1, 3, 2)
#plt.barh(labels, r2s, color='skyblue')
#plt.xlabel('R^2')
#plt.title('Model Performance (R^2)')

# C-Index
plt.subplot(1, 2, 2)
plt.barh(labels, c_indexes, color='purple')
plt.xlabel('Concordance Index')
plt.title('Model Performance (C-Index)')
# Set x-axis limits (starting from 15)
plt.xlim(0.5, max(c_indexes) + 0.05)  # Adjust the upper bound slightly for spacing

plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.show()