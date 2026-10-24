import os
from src.classes import CellGrowthModeler


def main():
    # Define path to the 4-factor cell growth dataset
    dataset_path = os.path.join("datasets", "dataset_cell_growth_4factor.csv")

    # Make the modeler class
    modeler = CellGrowthModeler(
        dataset_path=dataset_path,
        target_col="Specific Growth Rate [d^-1]",
        test_size=0.2,
        random_state=0,
        cv=4
    )

    # Stage 1: Load data, export summary stats, and split into train/test sets
    modeler.load_and_split_data()

    # Stage 2: Perform GridSearchCV hyperparameter optimization and plot validation curves
    modeler.optimize_polynomial_regression(max_degree=6)
    modeler.optimize_knn_regression(max_k=50)

    # Stage 3: Evaluate and compare selected models on the held-out test set
    modeler.evaluate_and_compare_models()

    # Stage 4: Generate learning curves across training set sample sizes
    modeler.generate_learning_curves()

    # Stage 5: Extract polynomial weights and plot 1D feature effect curves
    modeler.interpret_models_and_plot_effects()


if __name__ == "__main__":
    main()