import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np 

from sklearn.inspection import permutation_importance
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, cross_validate

class Toolbox: 
    def __init__(self,working_folder):
        self.working_folder = working_folder

    def makedir_soft(self) : 
        if not os.path.isdir(self.working_folder):
            os.makedirs(self.working_folder)


class PredictorOGTBase(Toolbox) : 
    def __init__(self, data, target, target_label, model, cv,working_folder, exp_name, groups=None):
        super().__init__(working_folder)
        self.data = data
        self.target_label = target_label
        self.target = np.ravel(target)
        self.target_pred = None 
        self.model = model
        self.model_hyperparams = {}
        self.cv = cv
        self.exp_name = exp_name
        self.groups = groups

        self.len_features = len(self.data.columns)

    def boxplot_target(self) : 
        sns.set_theme(style="ticks")
        f, ax = plt.subplots(figsize=(5,10))
        sns.boxplot(self.target)
        sns.stripplot(self.target,color=".3")
        ax.xaxis.grid(True)
        ax.set(ylabel="")

        if self.working_folder is not None : 
            ax.set(xlabel = self.target_label,title=f"{self.exp_name}")
            f.savefig(os.path.join(self.working_folder,"boxplot_target.png"))
            plt.close('all')

    def boxplot_data(self) : 
        sns.set_theme(style="ticks")
        f, ax = plt.subplots(figsize=(self.len_features*1.5,10))
        sns.boxplot(self.data)
        ax.xaxis.grid(True)
        ax.set(ylabel="")

        if self.working_folder is not None : 
            ax.set(xlabel = "Features",
                    ylabel="Frequencies (log scale)",
                    title=f"{self.exp_name}")
            f.savefig(os.path.join(self.working_folder,"boxplot_data.png"))
            plt.close('all')

    def histogram_data(self) : 
        ax = self.data.hist(figsize = (self.len_features,self.len_features))
        fig = ax[0][0].get_figure()

        if self.working_folder is not None : 
            fig.savefig(os.path.join(self.working_folder,"histogram_data.png"))
            plt.close('all')
        
    def show_splits(self) : 
        """
        Take data and target and show how cv split are made (brainstorm about visualization) 
        """
        pass

    def prediction_error_display(self) : 
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

        if self.working_folder is not None : 
            fig.savefig(os.path.join(self.working_folder,f"prediction_error_display_{self.exp_name}.png"))
            plt.close('all')

    def get_hyperparams(self,verbosity=False): 
        self.model_hyperparams = self.model.get_params()

        if verbosity : 
            for k,v in self.model_hyperparams.items() :
                print(k,";",v) 

        return self.model_hyperparams

    def set_hyperparams(self,model_hyperparams) : 
        self.model_hyperparams = model_hyperparams
        self.model.set_params(**model_hyperparams)

        return self.model_hyperparams
    
    def check_if_linear_regression(self) : 
        if self.model.__class__.__name__ == "Pipeline" : 
            if "LinearRegression" in [step.__class__.__name__ for step in self.model][-1] :
                return True
        elif self.model.__class__.__name__ == "LinearRegression" : 
            return True
        else : 
            return False 


class PredictorOGT(PredictorOGTBase) : 
    def __init__(self, data, target, target_label, model, cv, working_folder, exp_name, groups=None):
        super().__init__(data, target, target_label, model, cv, working_folder, exp_name, groups)
        self.cv_results = None
        self.results_models = None

    def run_visualisation_data(self) :
        self.makedir_soft() 
        self.boxplot_data()
        self.boxplot_target()
        self.histogram_data()
        plt.close('all')

    def search_hyperparams(self,param_grid,verbosity=False): 
        if param_grid is not None : 
            if self.groups is None : 
                search_cv = RandomizedSearchCV(
                    self.model,
                    param_distributions=param_grid,
                    scoring="neg_mean_absolute_error",
                    cv = self.cv,
                    n_iter=20,
                    random_state=42,
                    n_jobs=20
                    )

                search_cv.fit(self.data,self.target)
                self.get_hyperparams(verbosity)
                self.set_hyperparams(search_cv.best_params_)

                return search_cv.best_params_
        
            else : 
                    search_cv = RandomizedSearchCV(
                    self.model,
                    param_distributions=param_grid,
                    scoring="neg_mean_absolute_error",
                    cv = self.cv.split(self.data,self.target,groups = self.groups),
                    n_iter=20,
                    random_state=42,
                    n_jobs=20
                    )

                    search_cv.fit(self.data,self.target)
                    self.get_hyperparams(verbosity)
                    self.set_hyperparams(search_cv.best_params_)

            return search_cv.best_params_
        
        else : 
            self.get_hyperparams(verbosity)
            return self.model_hyperparams

    def predict_target(self) : 
        self.target_pred = cross_val_predict(self.model,self.data,self.target,cv=self.cv,groups=self.groups)
        self.prediction_error_display()

        return self.target_pred

    def estimate_error(self) : 
        self.cv_results = cross_validate(self.model,
                                         self.data,
                                         self.target,
                                         cv=self.cv,
                                         groups=self.groups,
                                         return_estimator=True,
                                         scoring=["neg_mean_absolute_error","r2"])

        return self.cv_results

    def permutation_importance_test(self) : 
        train_index, test_index = next(self.cv.split(self.data,groups=self.groups))

        # Créer les ensembles
        X_train = self.data.iloc[train_index,:]
        X_test = self.data.iloc[test_index,:]
        y_train = self.target[train_index]
        y_test = self.target[test_index]

        self.model.fit(X_train, y_train)
        fig, ax = plt.subplots(figsize=(7, 6))

        result = permutation_importance(self.model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=10)
        perm_sorted_idx = result.importances_mean.argsort()
        tick_labels_dict = {"tick_labels": X_test.columns[perm_sorted_idx]}
        ax.boxplot(result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict)
        ax.axvline(x=0, color="k", linestyle="--")

        ax.set_title(f"Permutation Importances of features (test set)\n Baseline accuracy on test data: {self.model.score(X_test, y_test):.2}\n {self.exp_name}")
        ax.set_xlabel("Decrease in accuracy score")
        _ = ax.figure.tight_layout()

        if self.working_folder is not None : 
            fig.savefig(os.path.join(self.working_folder,f"permutation_importance_{self.exp_name}.png"))
            plt.close('all')

    def plot_linear_coefs(self) : 
        if self.check_if_linear_regression() : 

            coefs = [pipeline[-1].coef_ for pipeline in self.cv_results["estimator"]]
            coefs = pd.DataFrame(coefs, columns=self.data.columns)
            sns.set_theme(style="ticks")
            f, ax = plt.subplots(figsize=(20,10))
            sns.boxplot(coefs)
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)

            if self.working_folder is not None : 
                ax.set(xlabel = "Features",
                    ylabel="Regression coefficients",
                    title=f"{self.exp_name}")
                f.savefig(os.path.join(self.working_folder,f"plot_linear_coef_{self.exp_name}.png")) 
                plt.close('all')

        else : 
            print("The model is not a LinearRegression, plot not drawn")

    def summarize_results(self) : 
            results_table = [] 
            results_table.append(self.exp_name)
            results_table.append(f"{-self.cv_results['test_neg_mean_absolute_error'].mean():.3f}")
            results_table.append(f"{self.cv_results['test_neg_mean_absolute_error'].std():.3f}")
            results_table.append(f"{self.cv_results['test_r2'].mean():.3f}")
            results_table.append(f"{self.cv_results['test_r2'].std():.3f}")
            self.results_models = pd.DataFrame([results_table], columns=["model","accuracy_mean","accuracy_std","standard_deviation_mean","standard_deviation_std"])
            self.results_models.to_csv(os.path.join(self.working_folder,f"{self.exp_name}_results.csv"), sep="\t")

            return self.results_models 