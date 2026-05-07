"""
Utilities to automatize ML analysis on a vector to predict OGT. 
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_predict,
    cross_validate,
)


class Toolbox:
    """
    Toolbox class to be herited by other classes

    Attribute :
    ------------
    working_folder : main folder where to store results

    Methods :
    ---------
    makedir_soft : Create directory without destroying current files inside
        if the directory already exists.
    """

    def __init__(self, working_folder):
        """
        Constructs all the necessary attributes for the Toolbox object.

        Parameter :
        -----------
        working_folder : main folder where to store results
        """
        self.working_folder = working_folder

    def makedir_soft(self):
        """
        Create directory without destroying current files inside if the directory already exists.
        """
        if not os.path.isdir(self.working_folder):
            os.makedirs(self.working_folder)


class PredictorOGTBase(Toolbox):
    """
    Class herited from Toolbox class.
    Manage all basic and common operations to initialize data for a machine learning
    process on a vector to predict OGT.

    Attributes :
    ------------
    data : Pandas DataFrame or array with the data to be trained and used
        to predict a target feature
    target : Pandas Series or array with the target data (here OGT)
    model : scikit-learn Pipeline object representing a machine-learning model to be tested
    cv : Cross-Validation splitter object (such as GroupKFold or ShuffleSplit, etc...)
    working_folder : main folder to store results
    exp_name : name of the ML experience for files and figures labelling.
    groups : optional, to set group with a list of groups when a Group-like CV is used.
    target_pred : different predictions of the target in all cross-validation folds.
    model_hyperparams : models hyperparameters chosen if relevant
    len_features : number of features used as input in data

    Methods :
    ---------
    boxplot_target : draw and save the boxplot to show the target distribution
    boxplot_data : draw and save boxplots to show the data distribution
    histogram_data : draw and save histograms to show the data distribution
    prediction_error_display : draw and save the prediction error plot for a given model
    get_hyperparams : print and returns hyperparameters used for the model
    set_hyperparams : take an input dictionary and set hyperparameters with chosen values
    check_if_linear_regression : check if the ML object LinearRegression is used or not.
    """

    def __init__(self, data, target, model, cv, working_folder, exp_name, groups=None):
        """
        Constructs all the necessary attributes for the PredictorOGTBase object.
        Herits all features from Toolbox.

        Parameters :
        ------------
        data : Pandas DataFrame or array with the data to be trained and used
            to predict a target feature
        target : Pandas Series or array with the target data (here OGT)
        model : scikit-learn Pipeline object representing a machine-learning model to be tested
        cv : Cross-Validation splitter object (such as GroupKFold or ShuffleSplit, etc...)
        working_folder : main folder to store results
        exp_name : name of the ML experience for files and figures labelling.
        groups : optional, to set group with a list of groups when a Group-like CV is used.
        """
        super().__init__(working_folder)
        self.data = data
        self.target = np.ravel(target)
        self.target_pred = None
        self.model = model
        self.model_hyperparams = {}
        self.cv = cv
        self.exp_name = exp_name
        self.groups = groups

        self.len_features = len(self.data.columns)

    def boxplot_target(self):
        """
        Draw and save the boxplot to show the target distribution.
        """
        sns.set_theme(style="ticks")
        f, ax = plt.subplots(figsize=(5, 10))
        sns.boxplot(self.target)
        sns.stripplot(self.target, color=".3")
        ax.xaxis.grid(True)
        ax.set(ylabel="")
        ax.set(
            xlabel="OGT", title=f"Boxplot of repartition for target ({self.exp_name})"
        )

        if self.working_folder is not None:
            f.savefig(os.path.join(self.working_folder, "boxplot_target.png"))

    def boxplot_data(self):
        """
        Draw and save boxplots to show the data distribution.
        """
        sns.set_theme(style="ticks")
        f, ax = plt.subplots(figsize=(self.len_features * 1.5, 10))
        sns.boxplot(self.data)
        ax.xaxis.grid(True)
        ax.set(ylabel="")
        ax.set(
            xlabel="Features",
            ylabel="Frequencies (log scale)",
            title=f"Boxplot of repartition for data ({self.exp_name})",
        )

        if self.working_folder is not None:
            f.savefig(os.path.join(self.working_folder, "boxplot_data.png"))

    def histogram_data(self):
        """
        Draw and save histograms to show the data distribution.
        """
        ax = self.data.hist(figsize=(self.len_features, self.len_features))
        fig = ax[0][0].get_figure()
        fig.suptitle("Histogram of values for data features", fontsize=25, y=0.92)

        if self.working_folder is not None:
            fig.savefig(os.path.join(self.working_folder, "histogram_data.png"))

    def prediction_error_display(self):
        """
        Draw and save the prediction error plot for a given model.
        """
        fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

        PredictionErrorDisplay.from_predictions(
            self.target,
            y_pred=self.target_pred,
            kind="actual_vs_predicted",
            ax=axs[0],
            random_state=0,
        )
        axs[0].set_title("Actual vs. Predicted values")

        PredictionErrorDisplay.from_predictions(
            self.target,
            y_pred=self.target_pred,
            kind="residual_vs_predicted",
            ax=axs[1],
            random_state=0,
        )
        axs[1].set_title("Residuals vs. Predicted Values")

        fig.suptitle(f"Plotting cross-validated predictions\n {self.exp_name}")
        plt.tight_layout()

        if self.working_folder is not None:
            fig.savefig(
                os.path.join(
                    self.working_folder, f"prediction_error_display_{self.exp_name}.png"
                )
            )
            plt.close("all")

    def get_hyperparams(self, verbosity=False):
        """
        Print and returns hyperparameters used for the model

        Parameter :
        -----------
        verbosity : True if hyperparameters have to be printed, else False (by default)

        Output :
        ---------
        model.hyperparams : hyperparameters of the model
        """
        self.model_hyperparams = self.model.get_params()

        if verbosity:
            for k, v in self.model_hyperparams.items():
                print(k, ";", v)

        return self.model_hyperparams

    def set_hyperparams(self, model_hyperparams):
        """
        Take an input dictionary and set hyperparameters with chosen values.
        No control is done on keys, so the user have to know the model.

        Parameters :
        ------------
        model_hyperparams : dictionary with hyperparameters to be set.

        Output :
        ---------
        model.hyperparams : hyperparameters of the model
        """
        self.model_hyperparams = model_hyperparams
        self.model.set_params(**model_hyperparams)

        return self.model_hyperparams

    def check_if_linear_regression(self):
        """
        Check if the ML object LinearRegression is used or not, as model itself or
        inside of a Pipeline object.

        Output :
        --------
        Boolean, True if the model is a LinearRegression, else False
        """
        if self.model.__class__.__name__ == "Pipeline":
            if (
                "LinearRegression"
                in [step.__class__.__name__ for step in self.model][-1]
            ):
                return True
        elif self.model.__class__.__name__ == "LinearRegression":
            return True
        else:
            return False


class PredictorOGT(PredictorOGTBase):
    """
    Final class herited from PredictorOGTBase with advanced methods to run and set models.

    Attributes :
    ------------
    cv_results : metrics computed with cross-validation to evaluate model quality
    results_models : DataFrame of cv_results

    Methods :
    ---------
    run_visualisation_data : call all data visualisation methods in PredictorOGTBase.
    search_hyperparams : With a RandomizedSearchCV, search the most optimized parameters set
        following a given param_grid, and implement them in the chosen model
    predict_target : predict the target with a cross-validation strategy and print the prediction
        error figure (absolute error and residues).
    estimate_error : compute metrics to evaluate model efficacity
    print_error_estimation : pretty prints differents metrics to evaluate the efficacity
        of the model
    permutation_importance_test : run a strategy of permutation importance test
        and displays the figure
    plot_linear_coefs : If the model is a LinearModel, plot the coefficient for each feature
    summarize_results : make a csv summary of metrics calculated by estimate_error method
    """

    def __init__(self, data, target, model, cv, working_folder, exp_name, groups=None):
        """
        Constructs all the necessary attributes for the PredictorOGT object.
        Herits all features from PredictorOGT.

        Parameters :
        ------------
        data : Pandas DataFrame or array with the data to be trained and used
            to predict a target feature
        target : Pandas Series or array with the target data (here OGT)
        model : scikit-learn Pipeline object representing a machine-learning model to be tested
        cv : Cross-Validation splitter object (such as GroupKFold or ShuffleSplit, etc...)
        working_folder : main folder to store results
        exp_name : name of the ML experience for files and figures labelling.
        groups : optional, to set group with a list of groups when a Group-like CV is used.
        """
        super().__init__(data, target, model, cv, working_folder, exp_name, groups)
        self.cv_results = None
        self.results_models = None

    def run_visualisation_data(self):
        """
        Call all data visualisation methods in PredictorOGTBase.
        """
        if self.working_folder is not None:
            self.makedir_soft()
        self.boxplot_data()
        self.boxplot_target()
        self.histogram_data()
        if self.working_folder is not None:
            plt.close("all")

    def search_hyperparams(self, param_grid, verbosity=False):
        """
        With a RandomizedSearchCV, search the most optimized parameters set
        following a given param_grid, and implement them in the chosen model.

        Parameters :
        ------------
        param_grid : grid of parameters to be tested
        verbosity : True if hyperparameters have to be printed, else False (by default)
        """
        if param_grid is not None:
            if self.groups is None:
                search_cv = RandomizedSearchCV(
                    self.model,
                    param_distributions=param_grid,
                    scoring="neg_mean_absolute_error",
                    cv=self.cv,
                    n_iter=20,
                    random_state=42,
                    n_jobs=20,
                )

                search_cv.fit(self.data, self.target)
                self.get_hyperparams(verbosity)
                self.set_hyperparams(search_cv.best_params_)

                return search_cv.best_params_

            else:
                search_cv = RandomizedSearchCV(
                    self.model,
                    param_distributions=param_grid,
                    scoring="neg_mean_absolute_error",
                    cv=self.cv.split(self.data, self.target, groups=self.groups),
                    n_iter=20,
                    random_state=42,
                    n_jobs=20,
                )

                search_cv.fit(self.data, self.target)
                self.get_hyperparams(verbosity)
                self.set_hyperparams(search_cv.best_params_)

            return search_cv.best_params_

        else:
            self.get_hyperparams(verbosity)
            return self.model_hyperparams

    def predict_target(self):
        """
        Predicts the target with a cross-validation strategy and print the prediction
        error figure (absolute error and residues).

        Output :
        --------
        self.target_pred : array with predicted target
        """
        self.target_pred = cross_val_predict(
            self.model, self.data, self.target, cv=self.cv, groups=self.groups
        )
        self.prediction_error_display()

        return self.target_pred

    def estimate_error(self):
        """
        Compute metrics to evaluate model efficacity, with cross_validate
        Efficacity metrics computed, for each folds are :
            - Fit time
            - Score time
            - MAE
            - R2

        Ouput :
        -------
        self.cv_results : dictionary with model efficacity metrics
        """
        self.cv_results = cross_validate(
            self.model,
            self.data,
            self.target,
            cv=self.cv,
            groups=self.groups,
            return_estimator=True,
            scoring=["neg_mean_absolute_error", "r2"],
        )

        return self.cv_results

    def print_error_estimation(self):
        """
        Pretty prints differents metrics to evaluate the efficacity of the model.
        Mean and STD of metrics are computed for all cross-validation folds before printing :
            - Fit time
            - Score time
            - MAE
            - R2
        """
        result = self.estimate_error()
        print(
            f"Mean fit time : {result["fit_time"].mean():.3f} +/- {result["fit_time"].std():.3f}"
        )
        print(
            f"Mean score time : {result["score_time"].mean():.3f} +/- {result["score_time"].std():.3f}"
        )
        print(
            f"Mean of error prediction for all CV splits : { - result["test_neg_mean_absolute_error"].mean():.3f} +/- {result["test_neg_mean_absolute_error"].std():.3f}"
        )
        print(
            f"Mean of r2 for all CV splits : {result["test_r2"].mean():.3f} +/- {result["test_r2"].std():.3f}"
        )

    def permutation_importance_test(self):
        """
        Run a strategy of permutation importance test, saves and displays the figure.
        """
        train_index, test_index = next(self.cv.split(self.data, groups=self.groups))

        # Créer les ensembles
        X_train = self.data.iloc[train_index, :]
        X_test = self.data.iloc[test_index, :]
        y_train = self.target[train_index]
        y_test = self.target[test_index]

        self.model.fit(X_train, y_train)
        fig, ax = plt.subplots(figsize=(7, 6))

        result = permutation_importance(
            self.model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=10
        )
        perm_sorted_idx = result.importances_mean.argsort()
        tick_labels_dict = {"tick_labels": X_test.columns[perm_sorted_idx]}
        ax.boxplot(
            result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict
        )
        ax.axvline(x=0, color="k", linestyle="--")

        ax.set_title(
            f"Permutation Importances of features (test set)\n Baseline accuracy on test data: {self.model.score(X_test, y_test):.2}\n {self.exp_name}"
        )
        ax.set_xlabel("Decrease in accuracy score")
        _ = ax.figure.tight_layout()

        if self.working_folder is not None:
            fig.savefig(
                os.path.join(
                    self.working_folder, f"permutation_importance_{self.exp_name}.png"
                )
            )
            plt.close("all")

    def plot_linear_coefs(self):
        """
        If the model is a LinearModel, plot and save coefficients for each feature.
        """
        if self.check_if_linear_regression():

            coefs = [pipeline[-1].coef_ for pipeline in self.cv_results["estimator"]]
            coefs = pd.DataFrame(coefs, columns=self.data.columns)
            sns.set_theme(style="ticks")
            f, ax = plt.subplots(figsize=(20, 10))
            sns.boxplot(coefs)
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)
            ax.set(
                xlabel="Features",
                ylabel="Regression coefficients",
                title=f"Boxplot of regression coefficients for each features ({self.exp_name})",
            )

            if self.working_folder is not None:
                f.savefig(
                    os.path.join(
                        self.working_folder, f"plot_linear_coef_{self.exp_name}.png"
                    )
                )
                plt.close("all")

        else:
            print("The model is not a LinearRegression, plot not drawn")

    def summarize_results(self):
        """
        Make a csv summary of metrics calculated by estimate_error method.

        Output :
        --------
        self.results_models : DataFrame with all metrics (Mean + STD) :
            - MAE
            - R2
        """
        results_table = []
        results_table.append(self.exp_name)
        results_table.append(
            f"{-self.cv_results['test_neg_mean_absolute_error'].mean():.3f}"
        )
        results_table.append(
            f"{self.cv_results['test_neg_mean_absolute_error'].std():.3f}"
        )
        results_table.append(f"{self.cv_results['test_r2'].mean():.3f}")
        results_table.append(f"{self.cv_results['test_r2'].std():.3f}")
        self.results_models = pd.DataFrame(
            [results_table],
            columns=[
                "model",
                "accuracy_mean",
                "accuracy_std",
                "standard_deviation_mean",
                "standard_deviation_std",
            ],
        )
        self.results_models.to_csv(
            os.path.join(self.working_folder, f"{self.exp_name}_results.csv"), sep="\t"
        )

        return self.results_models
