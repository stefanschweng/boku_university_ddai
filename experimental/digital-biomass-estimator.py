# %%  1. AI-based integration of data in local servers
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# %% Inspect data (data wrangler)
with open("data/biomass_sample_data_reps_separate.pkl", "rb") as f:
    df_loaded = pickle.load(f)
df_loaded

# %% Plot digital biomass of each cultivar over days of phenotyping
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_loaded,
    x="days_of_phenotyping",
    y="digital_biomass",
    hue="species",
    style="drought_stress",
    markers=True,
    dashes=False,
)
plt.title("Digital Biomass Over Days of Phenotyping by Species and Drought Stress")
plt.xlabel("Days of Phenotyping")
plt.ylabel("Digital Bi£omass")
plt.legend(title="Species and Drought Stress")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Train Ridge model

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold

# ---------------------
# 1. Load your dataset
# ---------------------
# Assuming your DataFrame is called `df`
# and contains the following columns:
# - day_of_phenotyping (int 1–20)
# - species (categorical: 4 species)
# - nitrogen_applied (float/int, only 2 values here)
# - drought_stress (bool or 0/1)
# - mean_plant_temperature, std_plant_temperature, median_plant_temperature,
#   min_plant_temperature, max_plant_temperature
# - digital_biomass (target variable)

# Example placeholder:
# df = pd.read_csv("your_data.csv")

# ----------------------
# 2. Define Features/Target
# ----------------------
feature_cols = [
    "days_of_phenotyping",
    "species",
    "nitrogen_applied",
    "drought_stress",
    # "mean_plant_temperature",
    # "std_plant_temperature",
    # "median_plant_temperature",
    # "min_plant_temperature",
    # "max_plant_temperature",
]
target_col = "digital_biomass"

X_data_init = df_loaded[feature_cols]
y_data_init = df_loaded[target_col]

# -----------------------
# 3. Define Column Groups
# -----------------------
spline_features = ["days_of_phenotyping"]
categorical_features = ["species"]
numeric_features = [
    "nitrogen_applied",
    "drought_stress",
    # "mean_plant_temperature",
    # "std_plant_temperature",
    # "median_plant_temperature",
    # "min_plant_temperature",
    # "max_plant_temperature",
]

# --------------------------
# 4. Build Preprocessing Pipeline
# --------------------------
preprocessor = ColumnTransformer(
    [
        ("spline_day", SplineTransformer(degree=3, n_knots=5, include_bias=False), spline_features),
        (
            "onehot_species",
            Pipeline(
                [
                    ("onehot", OneHotEncoder(drop="first")),
                    ("scaler", StandardScaler(with_mean=False)),  # Important: with_mean=False for sparse data
                ]
            ),
            categorical_features,
        ),
        ("num", StandardScaler(), numeric_features),
    ]
)

# %% Calculate Variance Inflation Factor (VIF) to check multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(X):
    """Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


X_data_init_processed = preprocessor.fit_transform(X_data_init)


# Convert to DataFrame with feature names
def get_feature_names(preprocessor, X_fit):
    if not hasattr(preprocessor, "transformers_"):
        preprocessor.fit(X_fit)
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "spline_day":
            n_out = trans.n_features_out_
            feature_names.extend([f"spline_day_{i}" for i in range(n_out)])
        elif name == "onehot_species":
            onehot = trans.named_steps["onehot"]
            cats = onehot.get_feature_names_out(cols)
            feature_names.extend(cats)
        elif name == "num":
            feature_names.extend(cols)
    return feature_names


X_data_init_processed = pd.DataFrame(
    X_data_init_processed, columns=get_feature_names(preprocessor, X_data_init), index=X_data_init.index
)
vif_df = calculate_vif(X_data_init_processed)
print("Variance Inflation Factor (VIF) for each feature:")
print(vif_df)

# %%

# --------------------------
# 5. Full Pipeline: Preprocessing + Model
# --------------------------

# Choose either Ridge or Lasso
model = Ridge()  #  LassoCV(cv=5, max_iter=10000, random_state=42)  # Ridge(alpha=1.0) # Lasso(alpha=0.1)
pipeline = Pipeline([("preprocess", preprocessor), ("regressor", model)])

param_grid = {"regressor__alpha": np.arange(0, 3, 0.05)}  # ridge regularization strength
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, scoring="r2")

# --------------------------
# 6. Train/Test Split
# --------------------------
# Create a stratification key combining species and nitrogen
df_loaded["strata"] = df_loaded["species"].astype(str) + "_" + df_loaded["nitrogen_applied"].astype(str)
strata = df_loaded["strata"]

X_train, X_test, y_train, y_test = train_test_split(
    X_data_init, y_data_init, test_size=0.3, random_state=42  #  , stratify=strata
)

y_test_avg = y_test.mean()
print("Average y_test:", y_test_avg)
# %%
# --------------------------
# 6b. Scale target variable (y)
# --------------------------
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# --------------------------
# 7. Fit Model
# --------------------------
grid.fit(X_train, y_train_scaled)

# --------------------------
# 8. Evaluate
# --------------------------

best_ridge_model = grid.best_estimator_.named_steps["regressor"]
# Predict scaled targets on test set
y_pred_scaled = grid.predict(X_test)

# Inverse transform predictions back to original scale
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Compute R² score on original scale
from sklearn.metrics import r2_score

print("Ridge Regression")
print(f"Best parameters: {grid.best_params_}")
r2_test_ridge = r2_score(y_test, y_pred)
rmse_test_ridge = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"Test R² score (best model, original scale): {r2_test_ridge:.3f}")
print(f"Test RMSE (best model, original scale): {rmse_test_ridge:.3e}")
print(f"Test NRMSE (best model, original scale): {(rmse_test_ridge / y_test_avg):.3%}")

# Optional: Cross-validation scoring on scaled y
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train_scaled, cv=5, scoring="r2", n_jobs=-1)
print(f"CV R² mean ± std (scaled y): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Plot predictions vs ground truth
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Ground Truth (digital_biomass)")
plt.ylabel("Predicted (digital_biomass)")
plt.title("Ridge Regression: Predictions vs. Ground Truth (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Demonstrate SplineTransformer
# Create the SplineTransformer
spline = SplineTransformer(degree=3, n_knots=5, include_bias=False)

days = np.arange(1, 21).reshape(-1, 1)

# Fit and transform the data
X_spline = spline.fit_transform(days)

for i in range(X_spline.shape[1]):
    plt.plot(days, X_spline[:, i], label=f"basis {i}")
plt.title("Spline Basis Functions")
plt.legend()
plt.show()

# %%

# Get coefficients for the spline basis functions from the trained Ridge model
coefs = grid.best_estimator_.named_steps["regressor"].coef_[
    : X_spline.shape[1]
]  # Spline features are first in the design matrix

# Compute model's spline contribution over input range
y_spline_curve = X_spline @ coefs

plt.figure(figsize=(8, 5))
plt.plot(days.flatten(), y_spline_curve, label="Model prediction (spline part)")
plt.scatter(X_train["days_of_phenotyping"], y_train_scaled, color="gray", alpha=0.5, label="Training data")
plt.title("Ridge Model Prediction Based on Day Spline Features")
plt.xlabel("Day")
plt.ylabel("Predicted Value (partial)")
plt.legend()
plt.show()

# %% SVR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Define SVR model
svr_model = SVR(kernel="rbf")
svr_pipeline = Pipeline([("preprocess", preprocessor), ("regressor", svr_model)])

# Define parameter grid for SVR
svr_param_grid = {
    "regressor__C": np.arange(0.1, 5, 0.1),  # Regularization parameter
    "regressor__kernel": ["rbf", "linear", "poly"],  # Kernel type
    "regressor__epsilon": np.arange(0.05, 0.4, 0.05),  # Epsilon in the epsilon-SVR model
    "regressor__gamma": ["scale", "auto"],
}

svr_grid = GridSearchCV(svr_pipeline, param_grid=svr_param_grid, cv=5, scoring="r2", n_jobs=-1)
svr_grid.fit(X_train, y_train_scaled)

# --------------------------
# 8. Evaluate
# --------------------------
# Predict scaled targets on test set
y_pred_scaled = svr_grid.predict(X_test)

# Inverse transform predictions back to original scale
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Compute R² score on original scale
from sklearn.metrics import r2_score

print("SVR")
print(f"Best SVR params: {svr_grid.best_params_}")
r2_test_svr = r2_score(y_test, y_pred)
rmse_test_svr = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"Test R² score (best model, original scale): {r2_test_svr:.3f}")
print(f"Test RMSE (best model, original scale): {rmse_test_svr:.3e}")
print(f"Test NRMSE (best model, original scale): {(rmse_test_svr / y_test_avg):.3%}")

# Optional: Cross-validation scoring on scaled y
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(svr_grid.best_estimator_, X_train, y_train_scaled, cv=5, scoring="r2", n_jobs=-1)
print(f"CV R² mean ± std (scaled y): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Predict on test set
y_pred = svr_grid.predict(X_test)
# Inverse transform predictions back to original scale
y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

# Plot predictions vs ground truth
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Ground Truth (digital_biomass)")
plt.ylabel("Predicted (digital_biomass)")
plt.title("SVR: Predictions vs. Ground Truth (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print number of support vectors
print("Number of training samples:", X_train.shape[0])
best_svr_model = svr_grid.best_estimator_.named_steps["regressor"]
print(f"Number of support vectors: {best_svr_model.support_.shape[0]}")
print("Percentage of support vectors: ", best_svr_model.support_.shape[0] / X_train.shape[0] * 100)

# %% SHAP values for SVR
import shap

X_background_svr = X_train.sample(100, random_state=0)
X_explain_svr = X_test.sample(10, random_state=1)


# predict_fn = lambda x: svr_grid.predict(x)
# Safe prediction function that reshapes if needed
def predict_fn_svr(X):
    # Ensure it's always a DataFrame, even for a single row
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    elif isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=X_background_svr.columns)
    # return best_svr_model.predict(X)  # For Individual species SHAP values
    return svr_grid.best_estimator_.predict(X)


# For Individual species SHAP values
# X_background_svr_processed = preprocessor.transform(X_background_svr)
# X_background_svr_processed = pd.DataFrame(
#     X_background_svr_processed, columns=get_feature_names(preprocessor, X_background_svr), index=X_background_svr.index
# )
# X_background_svr = X_background_svr_processed

explainer = shap.KernelExplainer(predict_fn_svr, X_background_svr)

# %% SHAP: global explanations
global_shap_values = explainer.shap_values(X_background_svr)
shap.summary_plot(global_shap_values, X_background_svr)
shap.summary_plot(global_shap_values, X_background_svr, plot_type="bar")

# If shap_values is a numpy array of shape (num_samples, num_features)
mean_abs_shap = np.abs(global_shap_values).mean(axis=0)

# Create a pandas Series with feature names and their importance
feature_importance = pd.Series(mean_abs_shap, index=X_background_svr.columns)

# Sort descending
feature_importance = feature_importance.sort_values(ascending=False)
print(feature_importance)

# %% SHAP: local explanations
local_shap_values = explainer.shap_values(X_explain_svr)
shap.summary_plot(local_shap_values, X_explain_svr)
shap.summary_plot(local_shap_values, X_explain_svr, plot_type="bar")
i = 1  # 0...9
print(f"SHAP values for test sample {i}:")
shap.force_plot(explainer.expected_value, local_shap_values[i], features=X_test.iloc[i], matplotlib=True)

# %% Random Forest
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Enable OOB in model
rf_model = RandomForestRegressor(random_state=42, oob_score=True)

rf_pipeline = Pipeline([("preprocess", preprocessor), ("regressor", rf_model)])

rf_param_grid = {
    "regressor__n_estimators": [10, 100, 1000],
    "regressor__max_depth": [None, 5, 10],
    "regressor__min_samples_split": [2, 3, 4, 5],
}

# Use GridSearchCV as before
rf_grid = GridSearchCV(rf_pipeline, param_grid=rf_param_grid, cv=5, scoring="r2", n_jobs=-1)
rf_grid.fit(X_train, y_train_scaled)

# Extract best estimator and model
best_rf_model = rf_grid.best_estimator_.named_steps["regressor"]

print("Random Forest")
# OOB score on training data
print(f"OOB R² score (best model): {best_rf_model.oob_score_:.3f}")

# Predict scaled targets on test set
y_pred_scaled = rf_grid.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
r2_test_rf = r2_score(y_test, y_pred)
rmse_test_rf = np.sqrt(np.mean((y_test - y_pred) ** 2))

print(f"Best RF params: {rf_grid.best_params_}")
print(f"Test R² score (best model, original scale): {r2_test_rf:.3f}")
print(f"Test RMSE (best model, original scale): {rmse_test_rf:.3e}")
print(f"Test NRMSE (best model, original scale): {(rmse_test_rf / y_test_avg):.3%}")

# Multiple CV runs with RepeatedKFold (e.g., 5 folds, 3 repeats)
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_scores = cross_val_score(rf_grid.best_estimator_, X_train, y_train_scaled, cv=rkf, scoring="r2", n_jobs=-1)

print(f"Repeated CV R² mean ± std (scaled y): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

######

# Plot predictions vs ground truth
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Ground Truth (digital_biomass)")
plt.ylabel("Predicted (digital_biomass)")
plt.title("Random Forest: Predictions vs. Ground Truth (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Calculate feature importances from the best Random Forest model
importances = best_rf_model.feature_importances_
feature_names = get_feature_names(preprocessor, X_data_init)

# Visualize feature importances
import matplotlib.pyplot as plt

indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
plt.tight_layout()
plt.show()

# %% SHAP values for RF
X_background_rf = X_train.sample(100, random_state=0)
X_explain_rf = X_test.sample(10, random_state=1)


# predict_fn = lambda x: svr_grid.predict(x)
# Safe prediction function that reshapes if needed
def predict_fn_RF(X):
    # Ensure it's always a DataFrame, even for a single row
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    elif isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=X_background_rf.columns)
    return rf_grid.best_estimator_.predict(X)


explainer_RF = shap.KernelExplainer(predict_fn_RF, X_background_rf)

# %% SHAP: global explanations
global_shap_values_RF = explainer_RF.shap_values(X_background_rf)
shap.summary_plot(global_shap_values_RF, X_background_rf)
shap.summary_plot(global_shap_values_RF, X_background_rf, plot_type="bar")

# If shap_values is a numpy array of shape (num_samples, num_features)
mean_abs_shap_RF = np.abs(global_shap_values_RF).mean(axis=0)

# Create a pandas Series with feature names and their importance
feature_importance_RF = pd.Series(mean_abs_shap_RF, index=X_background_rf.columns)

# Sort descending
feature_importance_RF = feature_importance_RF.sort_values(ascending=False)
print(feature_importance_RF)

# %% Plot model performance metrics
num_models = 3
plt.figure(figsize=(10, 6))
plt.title("R2 Scores")
plt.bar(range(num_models), [r2_test_ridge, r2_test_svr, r2_test_rf], align="center")
plt.xticks(range(num_models), ["Ridge", "SVR", "RF"], rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.title("RMSE Scores")
plt.bar(range(num_models), [rmse_test_ridge, rmse_test_svr, rmse_test_rf], align="center")
plt.xticks(range(num_models), ["Ridge", "SVR", "RF"], rotation=90)
plt.tight_layout()
plt.show()

# %% Test predictions on new data

# Features
#    "days_of_phenotyping",
#     "species",
#     "nitrogen_applied",
#     "drought_stress",

# Example new data point
# Create new test dataframe:

test_species = "barley_1"  # "durum" or "barley_2", "barley_1", "barley_3"
test_nitrogen = 25  # 130 or 25
test_drought_stress = 1  # 0 or 1

test_days = np.arange(1, 21)
new_test_df = pd.DataFrame(
    {
        "days_of_phenotyping": test_days,
        "species": [test_species] * len(test_days),
        "nitrogen_applied": [test_nitrogen] * len(test_days),
        "drought_stress": [test_drought_stress] * len(test_days),
    }
)

# Preprocess new data
preprocessed_new = preprocessor.transform(new_test_df)

# Predict with all 3 models
ridge_pred_scaled = best_ridge_model.predict(preprocessed_new)
svr_pred_scaled = best_svr_model.predict(preprocessed_new)
rf_pred_scaled = best_rf_model.predict(preprocessed_new)

# Inverse transform to original scale
ridge_pred = y_scaler.inverse_transform(ridge_pred_scaled.reshape(-1, 1)).ravel()
svr_pred = y_scaler.inverse_transform(svr_pred_scaled.reshape(-1, 1)).ravel()
rf_pred = y_scaler.inverse_transform(rf_pred_scaled.reshape(-1, 1)).ravel()

# Get ground truth for barley_2, 130 N, 0 drought
ground_truth = df_loaded[
    (df_loaded["species"] == test_species)
    & (df_loaded["nitrogen_applied"] == test_nitrogen)
    & (df_loaded["drought_stress"] == test_drought_stress)
].sort_values("days_of_phenotyping")

# Visualize predictions and ground truth (only plot ground truth if it exists)
plt.figure(figsize=(10, 6))
plt.plot(test_days, ridge_pred, label="Ridge", marker="o")
plt.plot(test_days, svr_pred, label="SVR", marker="s")
plt.plot(test_days, rf_pred, label="Random Forest", marker="^")
if not ground_truth.empty:
    plt.plot(
        ground_truth["days_of_phenotyping"],
        ground_truth["digital_biomass"],
        label="Ground Truth",
        marker="x",
        linestyle="--",
        color="black",
    )
plt.xlabel("Days of Phenotyping")
plt.ylabel("Predicted Digital Biomass (mm³)")
# plt.title(f"Predicted Digital Biomass for {test_species} (N={test_nitrogen}, Drought={test_drought_stress})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% Investigate model predictions over days for different nitrogen levels
def plot_model_predictions_over_days(
    test_days,
    test_species,
    test_drought_stress,
    test_nitrogen_levels,
    preprocessor,
    y_scaler,
    model,
    model_name,
    integrated_data,
    ax=None,
):
    """
    Plot model predictions over days for different nitrogen levels, with ground truth if available.
    Plots on the given matplotlib axis.
    """
    show_plt = False
    if ax is None:
        show_plt = True
        fig, ax = plt.subplots(figsize=(10, 6))

    # Define a color palette with high contrast for up to 4 lines
    color_palette = ["#0072B2", "#D50000"]  # black, orange, blue, dark orange

    for idx, n_val in enumerate(test_nitrogen_levels):
        # Plot ground truth if available
        ground_truth = integrated_data[
            (integrated_data["species"] == test_species)
            & (integrated_data["nitrogen_applied"] == n_val)
            & (integrated_data["drought_stress"] == test_drought_stress)
        ].sort_values("days_of_phenotyping")
        if not ground_truth.empty:
            ax.plot(
                ground_truth["days_of_phenotyping"],
                ground_truth["digital_biomass"],
                marker="x",
                linestyle="--",
                color=color_palette[idx % len(color_palette)],
                label=f"GT N={n_val}",
                alpha=0.7,
                linewidth=2,
                markersize=8,
                markeredgewidth=2,
            )

    color_palette = ["#FFA500", "#000000"]  # black, orange, blue, dark orange
    for idx, n_val in enumerate(test_nitrogen_levels):
        test_df = pd.DataFrame(
            {
                "days_of_phenotyping": test_days,
                "species": [test_species] * len(test_days),
                "nitrogen_applied": [n_val] * len(test_days),
                "drought_stress": [test_drought_stress] * len(test_days),
            }
        )
        preprocessed = preprocessor.transform(test_df)
        pred_scaled = model.predict(preprocessed)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        ax.plot(
            test_days,
            pred,
            marker="o",
            color=color_palette[idx % len(color_palette)],
            label=f"N {'<=' if n_val <= 76.958 else '>'} α",
        )
        # 76.958

    if test_species == "barley_2":
        test_species_str = "Barley v2"
    elif test_species == "durum":
        test_species_str = "Durum Wheat"
    elif test_species == "barley_1":
        test_species_str = "Barley v1"
    elif test_species == "bread_wheat":
        test_species_str = "Bread Wheat"

    ax.set_xlabel("Days of Phenotyping")
    ax.set_ylabel("Predicted Digital Biomass (mm³)")
    ax.set_title(f"{test_species_str} ({'Drought Condition' if test_drought_stress else 'Standard Condition'})")
    ax.grid(True)
    ax.legend()

    if show_plt:
        plt.tight_layout()
        plt.show()


# %% Test predictions over days for different nitrogen levels

test_species = "barley_2"
test_drought_stress = 0
test_days = np.arange(1, 21)
test_nitrogen_levels = np.arange(10, 176, 15)

# Plot predictions for Ridge model
# plot_model_predictions_over_days(
#     test_days,
#     test_species,
#     test_drought_stress,
#     test_nitrogen_levels,
#     preprocessor,
#     y_scaler,
#     best_ridge_model,
#     "Ridge Regression",
#     integrated_data,
# )

# Plot predictions for SVR model
plot_model_predictions_over_days(
    test_days,
    test_species,
    test_drought_stress,
    test_nitrogen_levels,
    preprocessor,
    y_scaler,
    best_svr_model,
    "SVR",
    df_loaded,
)

# %%

test_nitrogen_levels = np.arange(10, 176, 15)
# Plot predictions for Random Forest model
plot_model_predictions_over_days(
    test_days,
    test_species,
    test_drought_stress,
    [25, 130],
    preprocessor,
    y_scaler,
    best_rf_model,
    "Random Forest",
    df_loaded,
)

# %% Plot 2x2 subplots

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.ravel()

# Example 1
plot_model_predictions_over_days(
    test_days=np.arange(1, 21),
    test_species="barley_1",
    test_drought_stress=1,
    test_nitrogen_levels=np.arange(10, 100, 30),
    preprocessor=preprocessor,
    y_scaler=y_scaler,
    model=best_svr_model,
    model_name="SVR",
    integrated_data=df_loaded,
    ax=axs[0],
)

# Example 2
plot_model_predictions_over_days(
    test_days=np.arange(1, 21),
    test_species="durum",
    test_drought_stress=1,
    test_nitrogen_levels=np.arange(10, 80, 15),
    preprocessor=preprocessor,
    y_scaler=y_scaler,
    model=best_svr_model,
    model_name="SVR",
    integrated_data=df_loaded,
    ax=axs[1],
)

# Example 3
plot_model_predictions_over_days(
    test_days=np.arange(1, 21),
    test_species="barley_2",
    test_drought_stress=0,
    test_nitrogen_levels=np.arange(100, 176, 15),
    preprocessor=preprocessor,
    y_scaler=y_scaler,
    model=best_svr_model,
    model_name="SVR",
    integrated_data=df_loaded,
    ax=axs[2],
)

# Example 4
plot_model_predictions_over_days(
    test_days=np.arange(1, 21),
    test_species="bread_wheat",
    test_drought_stress=0,
    test_nitrogen_levels=np.arange(100, 176, 15),
    preprocessor=preprocessor,
    y_scaler=y_scaler,
    model=best_svr_model,
    model_name="SVR",
    integrated_data=df_loaded,
    ax=axs[3],
)

plt.tight_layout()
plt.show()

# %% Optimize for nitrogen levels
from scipy.optimize import minimize_scalar

test_species = "durum"
test_drought_stress = 1
day_to_optimize = 3


def objective(nitrogen_level):
    new_test_df = pd.DataFrame(
        {
            "days_of_phenotyping": [day_to_optimize],
            "species": [test_species],
            "nitrogen_applied": [nitrogen_level],
            "drought_stress": [test_drought_stress],
        }
    )

    # Preprocess new data
    preprocessed_new = preprocessor.transform(new_test_df)
    return -best_svr_model.predict(preprocessed_new)  # maximize biomass = minimize negative


res = minimize_scalar(objective, bounds=(0, 300), method="bounded")
best_nitrogen = res.x
print(
    f"Optimal nitrogen level for {test_species} on day {day_to_optimize} with drought stress {test_drought_stress}: {(best_nitrogen[0]):.2f} kg/ha"
)

# %%
