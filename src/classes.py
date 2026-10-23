import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# By default, matplotlib makes math in italics, but we want our units in regular.
plt.rcParams['mathtext.default'] = 'regular'


class CellGrowthModeler:
    """
    A class to load, preprocess, optimize, evaluate, and interpret machine learning
    models (Polynomial Regression and k-Nearest Neighbors) for predicting specific
    cell growth rates from bioprocess conditions.
    """

    # Plotting constants following course conventions
    FIG_WIDTH = 14.39 / 2.54
    FIG_HEIGHT = 12.09 / 2.54
    FONTSIZE_XY_LABEL = 14
    FONTSIZE_TICKS = 12
    FONTSIZE_LEGEND = 11
    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    STR_DM1 = r"$d^{-1}$"

    def __init__(self, dataset_path, target_col="Specific Growth Rate [d^-1]",
                 test_size=0.2, random_state=0, cv=4):
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.kwargs_gscv = dict(
            cv=self.cv,
            scoring="neg_root_mean_squared_error",
            return_train_score=True
        )

        # Ensure output directories exist
        os.makedirs("figures", exist_ok=True)
        os.makedirs("tables", exist_ok=True)

        # Data attributes (populated in load_and_split_data)
        self.df = None
        self.features = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # Model and results attributes
        self.model_gscv_poly = None
        self.model_gscv_knn = None
        self.df_cv_poly = None
        self.df_cv_knn = None

    def load_and_split_data(self):
        """
         Stage 1: Loads the dataset, separates features (X) and target (y),
        exports descriptive statistics, and splits data into training and test sets.
        """
        self.df = pd.read_csv(self.dataset_path)

        # Extract feature columns (all columns except the target column)
        self.features = [col for col in self.df.columns if col != self.target_col]

        self.x = self.df.loc[:, self.features].to_numpy()
        self.y = self.df.loc[:, self.target_col].to_numpy()

        # Split data into 80% training/validation and 20% unseen test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # Export dataset summary statistics and extreme growth conditions
        df_summary = self.df.describe().T.round(4)
        df_summary.to_csv(os.path.join("tables", "table_01_dataset_summary.csv"))

        df_sorted = self.df.sort_values(by=self.target_col, ascending=False)
        df_sorted.head(10).to_csv(os.path.join("tables", "table_06_high_growth_conditions.csv"), index=False)
        df_sorted.tail(10).to_csv(os.path.join("tables", "table_07_low_growth_conditions.csv"), index=False)

        print("\n=== Stage 1: Dataset Summary Statistics ===")
        print(df_summary.to_string())
        print()

    def optimize_polynomial_regression(self, max_degree=6):
        """
        Stage 2a: Optimizes Polynomial Regression degree using GridSearchCV,
        exports CV results to CSV, and plots the validation curve.
        """
        model_poly = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(include_bias=False),
            LinearRegression()
        )
        param_grid_poly = dict(
            polynomialfeatures__degree=[i for i in range(1, max_degree + 1)]
        )

        self.model_gscv_poly = GridSearchCV(
            model_poly, param_grid=param_grid_poly, **self.kwargs_gscv
        )
        self.model_gscv_poly.fit(self.x_train, self.y_train)
        self.df_cv_poly = pd.DataFrame(self.model_gscv_poly.cv_results_)

        # Format clean table with positive RMSE values
        df_export = pd.DataFrame({
            "rank": self.df_cv_poly["rank_test_score"],
            "degree": self.df_cv_poly["param_polynomialfeatures__degree"],
            "valid_rmse_mean": (-self.df_cv_poly["mean_test_score"]).round(5),
            "valid_rmse_std": self.df_cv_poly["std_test_score"].round(5),
            "train_rmse_mean": (-self.df_cv_poly["mean_train_score"]).round(5),
            "train_rmse_std": self.df_cv_poly["std_train_score"].round(5),
        })
        df_export.to_csv(os.path.join("tables", "table_02_cv_results_poly.csv"), index=False)

        print("=== Stage 2a: Polynomial Regression GridSearchCV Results ===")
        print(df_export.to_string(index=False))
        print()

        # Plot Validation Curve (degrees 1 to 4 to clearly show underfitting, optimal, and overfitting)
        plot_n = min(4, max_degree)
        degrees = self.df_cv_poly.loc[:, "param_polynomialfeatures__degree"].to_numpy()
        rmse_train = -self.df_cv_poly.loc[:, "mean_train_score"].to_numpy()
        rmse_valid = -self.df_cv_poly.loc[:, "mean_test_score"].to_numpy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(self.FIG_WIDTH, self.FIG_HEIGHT),
                               dpi=200, layout="constrained")
        ax.plot(degrees[:plot_n], rmse_train[:plot_n], label="Train (Poly)",
                color=self.COLORS[0], linestyle="-", marker="o")
        ax.plot(degrees[:plot_n], rmse_valid[:plot_n], label="Valid (Poly)",
                color=self.COLORS[0], linestyle="--", marker="o")
        ax.set_xlabel("Polynomial Degree", fontsize=self.FONTSIZE_XY_LABEL)
        ax.set_ylabel(f"Specific Growth Rate RMSE [{self.STR_DM1}]", fontsize=self.FONTSIZE_XY_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=self.FONTSIZE_TICKS)
        ax.legend(ncols=1, fontsize=self.FONTSIZE_LEGEND)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MultipleLocator(1))

        fig.savefig(os.path.join("figures", "figure_01_validation_curve_poly.png"))
        plt.close(fig)

    def optimize_knn_regression(self, max_k=50):
        """
        Stage 2b: Optimizes k-Nearest Neighbors (kNN) hyperparameters (n_neighbors and weights)
        using GridSearchCV, exports CV results to CSV, and plots the validation curve.
        """
        model_knn = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor()
        )
        param_grid_knn = dict(
            kneighborsregressor__n_neighbors=[i for i in range(1, max_k + 1)],
            kneighborsregressor__weights=["uniform", "distance"]
        )

        self.model_gscv_knn = GridSearchCV(
            model_knn, param_grid=param_grid_knn, **self.kwargs_gscv
        )
        self.model_gscv_knn.fit(self.x_train, self.y_train)
        self.df_cv_knn = pd.DataFrame(self.model_gscv_knn.cv_results_)

        df_sorted = self.df_cv_knn.sort_values(by=["rank_test_score"], ascending=True)
        df_export = pd.DataFrame({
            "rank": df_sorted["rank_test_score"],
            "k": df_sorted["param_kneighborsregressor__n_neighbors"],
            "weights": df_sorted["param_kneighborsregressor__weights"],
            "valid_rmse_mean": (-df_sorted["mean_test_score"]).round(5),
            "valid_rmse_std": df_sorted["std_test_score"].round(5),
            "train_rmse_mean": (-df_sorted["mean_train_score"]).round(5),
            "train_rmse_std": df_sorted["std_train_score"].round(5),
        })
        df_export.to_csv(os.path.join("tables", "table_03_cv_results_knn.csv"), index=False)

        print("=== Stage 2b: Top 10 kNN Regression GridSearchCV Results ===")
        print(df_export.head(10).to_string(index=False))
        print()

        # Plot Validation Curve for kNN
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(self.FIG_WIDTH, self.FIG_HEIGHT),
                               dpi=200, layout="constrained")

        for i, weight in enumerate(param_grid_knn["kneighborsregressor__weights"], start=1):
            mask_weight = (self.df_cv_knn.loc[:, "param_kneighborsregressor__weights"] == weight)
            df_sub = self.df_cv_knn.loc[mask_weight, :].sort_values(
                by=["param_kneighborsregressor__n_neighbors"], ascending=True
            )
            n_neighbors = df_sub.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy()
            rmse_train = -df_sub.loc[:, "mean_train_score"].to_numpy()
            rmse_valid = -df_sub.loc[:, "mean_test_score"].to_numpy()

            ax.plot(n_neighbors, rmse_train, label=f"Train (kNN: {weight})",
                    color=self.COLORS[i], linestyle="-")
            ax.plot(n_neighbors, rmse_valid, label=f"Valid (kNN: {weight})",
                    color=self.COLORS[i], linestyle="--")

        ax.set_xlabel("k", fontsize=self.FONTSIZE_XY_LABEL)
        ax.set_ylabel(f"Specific Growth Rate RMSE [{self.STR_DM1}]", fontsize=self.FONTSIZE_XY_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=self.FONTSIZE_TICKS)
        ax.legend(ncols=1, fontsize=self.FONTSIZE_LEGEND)
        ax.set_ylim(bottom=0)

        fig.savefig(os.path.join("figures", "figure_02_validation_curve_knn.png"))
        plt.close(fig)

    def evaluate_and_compare_models(self):
        """
        Stage 3: Evaluates the selected optimal model and reasonable alternatives
        from Polynomial and kNN families on the held-out test set.
        """
        best_deg = self.model_gscv_poly.best_params_["polynomialfeatures__degree"]
        best_k = self.model_gscv_knn.best_params_["kneighborsregressor__n_neighbors"]
        best_w = self.model_gscv_knn.best_params_["kneighborsregressor__weights"]

        models_to_compare = [
            ("Polynomial (degree=1, Linear)",
             make_pipeline(StandardScaler(), PolynomialFeatures(degree=1, include_bias=False), LinearRegression())),
            (f"Polynomial (degree={best_deg}, Optimal)",
             make_pipeline(StandardScaler(), PolynomialFeatures(degree=best_deg, include_bias=False), LinearRegression())),
            ("Polynomial (degree=3, Higher-Order)",
             make_pipeline(StandardScaler(), PolynomialFeatures(degree=3, include_bias=False), LinearRegression())),
            (f"kNN (k={best_k}, weights=distance)",
             make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=best_k, weights="distance"))),
            (f"kNN (k={best_k}, weights=uniform)",
             make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=best_k, weights="uniform"))),
        ]

        rows = []
        for name, model in models_to_compare:
            model.fit(self.x_train, self.y_train)
            y_tr_pred = model.predict(self.x_train)
            y_te_pred = model.predict(self.x_test)

            rows.append({
                "Model_Architecture": name,
                "Train_RMSE": round(root_mean_squared_error(self.y_train, y_tr_pred), 5),
                "Test_RMSE": round(root_mean_squared_error(self.y_test, y_te_pred), 5),
                "Train_R2": round(r2_score(self.y_train, y_tr_pred), 4),
                "Test_R2": round(r2_score(self.y_test, y_te_pred), 4)
            })

        df_comp = pd.DataFrame(rows)
        df_comp.to_csv(os.path.join("tables", "table_04_model_comparison_test.csv"), index=False)

        print("=== Stage 3: Final Model Comparison on Unseen Test Set ===")
        print(df_comp.to_string(index=False))
        print()

    def generate_learning_curves(self):
        """
        Stage 4: Generates learning curves by training GridSearchCV models on
        progressively larger random subsets of the dataset.
        """
        rng = np.random.default_rng(seed=1)

        lc_n_training_samples = []
        lc_rmse_poly_train = []
        lc_rmse_poly_test = []
        lc_rmse_knn_train = []
        lc_rmse_knn_test = []

        model_poly = make_pipeline(
            StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression()
        )
        param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, 7)])
        model_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())

        for n_samples in range(40, self.x.shape[0] + 1, 35):
            idxs = rng.choice(self.x.shape[0], size=n_samples, replace=False)
            x_sub = self.x[idxs]
            y_sub = self.y[idxs]

            x_tr, x_te, y_tr, y_te = train_test_split(
                x_sub, y_sub, test_size=self.test_size, random_state=self.random_state
            )

            sub_max_k = min(30, int(x_tr.shape[0] * 0.7))
            param_grid_knn = dict(
                kneighborsregressor__n_neighbors=[i for i in range(1, sub_max_k + 1)],
                kneighborsregressor__weights=["uniform", "distance"]
            )

            gscv_poly = GridSearchCV(model_poly, param_grid=param_grid_poly, **self.kwargs_gscv)
            gscv_poly.fit(x_tr, y_tr)

            gscv_knn = GridSearchCV(model_knn, param_grid=param_grid_knn, **self.kwargs_gscv)
            gscv_knn.fit(x_tr, y_tr)

            lc_n_training_samples.append(x_tr.shape[0])
            lc_rmse_poly_train.append(root_mean_squared_error(y_tr, gscv_poly.predict(x_tr)))
            lc_rmse_poly_test.append(root_mean_squared_error(y_te, gscv_poly.predict(x_te)))
            lc_rmse_knn_train.append(root_mean_squared_error(y_tr, gscv_knn.predict(x_tr)))
            lc_rmse_knn_test.append(root_mean_squared_error(y_te, gscv_knn.predict(x_te)))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(self.FIG_WIDTH, self.FIG_HEIGHT),
                               dpi=200, layout="constrained")
        ax.plot(lc_n_training_samples, lc_rmse_poly_train, label="RMSE Train (Poly)",
                color=self.COLORS[0], linestyle="-", marker="o")
        ax.plot(lc_n_training_samples, lc_rmse_poly_test, label="RMSE Test (Poly)",
                color=self.COLORS[0], linestyle="--", marker="o")
        ax.plot(lc_n_training_samples, lc_rmse_knn_train, label="RMSE Train (kNN)",
                color=self.COLORS[1], linestyle="-", marker="s")
        ax.plot(lc_n_training_samples, lc_rmse_knn_test, label="RMSE Test (kNN)",
                color=self.COLORS[1], linestyle="--", marker="s")

        ax.set_xlabel("Number of training samples", fontsize=self.FONTSIZE_XY_LABEL)
        ax.set_ylabel(f"Specific Growth Rate RMSE [{self.STR_DM1}]", fontsize=self.FONTSIZE_XY_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=self.FONTSIZE_TICKS)
        ax.legend(ncols=1, fontsize=self.FONTSIZE_LEGEND)
        ax.set_ylim(bottom=0)

        fig.savefig(os.path.join("figures", "figure_03_learning_curve.png"))
        plt.close(fig)

    def interpret_models_and_plot_effects(self):
        """
        Stage 5: Extracts standardized polynomial weights to interpret feature importance,
        and generates 1D sensitivity plots holding other features at their mean values.
        """
        best_deg = self.model_gscv_poly.best_params_["polynomialfeatures__degree"]
        best_k = self.model_gscv_knn.best_params_["kneighborsregressor__n_neighbors"]

        # Fit 2nd-order polynomial on full dataset to extract standardized weights
        model_poly_full = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=best_deg, include_bias=False),
            LinearRegression()
        )
        model_poly_full.fit(self.x, self.y)

        powers = model_poly_full["polynomialfeatures"].powers_
        coefs = model_poly_full["linearregression"].coef_
        intercept = model_poly_full["linearregression"].intercept_

        print(f"=== Stage 5: Standardized Polynomial Model Weights (Degree = {best_deg}) ===")
        print(f"{self.target_col} = ")
        print(f"{intercept:8.4f} (Baseline / Intercept)")

        terms_list = [{"Term": "Intercept (Baseline)", "Standardized_Weight": round(float(intercept), 5)}]

        for i in range(powers.shape[0]):
            partial_terms = []
            for j in range(powers.shape[1]):
                feature_unit_splitter = " ["
                if feature_unit_splitter in self.features[j]:
                    feature_name = self.features[j].split(feature_unit_splitter)[0]
                else:
                    feature_name = self.features[j]

                if powers[i, j] == 1:
                    partial_terms.append(f"[{feature_name}]")
                elif powers[i, j] > 1:
                    partial_terms.append(f"[{feature_name}]^{powers[i, j]}")

            coef = coefs[i]
            term = " * ".join(partial_terms)
            print(f"{coef:+8.4f} * {term}")
            terms_list.append({"Term": term, "Standardized_Weight": round(float(coef), 5)})

        print()
        df_weights = pd.DataFrame(terms_list)
        df_weights.to_csv(os.path.join("tables", "table_05_polynomial_weights.csv"), index=False)

        # Fit kNN models (distance and uniform) on full dataset for visual comparison
        model_knn_dist = make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=best_k, weights="distance")
        )
        model_knn_uni = make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=best_k, weights="uniform")
        )
        model_knn_dist.fit(self.x, self.y)
        model_knn_uni.fit(self.x, self.y)

        # Plot 1D sensitivity curves (holding other 3 features constant at their mean)
        x_means = np.mean(self.x, axis=0)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(self.FIG_WIDTH * 1.8, self.FIG_HEIGHT * 1.6),
                                 dpi=200, layout="constrained")
        axes_flat = axes.flatten()

        for j, feat_name in enumerate(self.features):
            ax = axes_flat[j]
            f_min = np.min(self.x[:, j])
            f_max = np.max(self.x[:, j])
            f_buffer = 0.12 * (f_max - f_min)

            # Avoid negative concentrations when plotting Glucose and Lactate
            low_bound = max(0.0, f_min - f_buffer) if "[mM]" in feat_name else f_min - f_buffer
            f_vals = np.linspace(low_bound, f_max + f_buffer, num=200)

            x_synth = np.tile(x_means, (200, 1))
            x_synth[:, j] = f_vals

            ax.scatter(self.x[:, j], self.y, color="grey", alpha=0.25, s=18,
                       edgecolors="none", label="Dataset samples", zorder=1)
            ax.plot(f_vals, model_poly_full.predict(x_synth), color=self.COLORS[0],
                    linestyle="-", linewidth=2.2, label=f"Poly (deg={best_deg})", zorder=3)
            ax.plot(f_vals, model_knn_dist.predict(x_synth), color=self.COLORS[1],
                    linestyle="--", linewidth=2, label=f"kNN (k={best_k}, distance)", zorder=2)
            ax.plot(f_vals, model_knn_uni.predict(x_synth), color=self.COLORS[2],
                    linestyle="-.", linewidth=2, label=f"kNN (k={best_k}, uniform)", zorder=2)

            ax.axvline(x=f_min, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
            ax.axvline(x=f_max, color="black", linestyle=":", linewidth=1.0, alpha=0.5,
                       label="Training bounds")
            ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3, zorder=0)

            ax.set_xlabel(feat_name, fontsize=self.FONTSIZE_XY_LABEL)
            ax.set_ylabel(f"Specific Growth Rate [{self.STR_DM1}]", fontsize=self.FONTSIZE_XY_LABEL)
            ax.tick_params(axis='both', which='major', labelsize=self.FONTSIZE_TICKS)
            if j == 0:
                ax.legend(loc="upper left", fontsize=9)

        fig.savefig(os.path.join("figures", "figure_04_feature_effects.png"))
        plt.close(fig)