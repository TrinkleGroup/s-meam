import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class SplineMEAM(BaseEstimator, RegressorMixin):

    """
    A wrapper for constructing a model. Designed to help interface the fitting
    process with sklearn functions (e.g. AdaBoostRegressor).

    NOTE: it's assumed that AdaBoostRegressor only runs on the master process
    """

    def __init__(self, knot_values=None, runParameters=None, template=None,
            node_manager=None):

        self.knot_values = knot_values
        self.runParameters = runParameters
        self.template = template
        self.node_manager = node_manager

    def fit(self, X, Y):
        """This should fit a potential and return itself (this SplineMEAM
        object, the fitted potential)."""

        pass

    # def get_params(self):
    #     """Get parameters for the estimator"""
    #     pass
    # 
    # def set_params(self):
    #     """Set parameters for the estimator"""
    #     pass

    def predict(self, X):
        """
        Predict regression value for X.

        Args:
            X: (np.arr) training samples; shape = (n_samples, ?)
                NOTE: you can figure out later how this would work for your data

        Returns:
            y: (np.arr) the predicted regression values; shape = (n_samples)
        """
        pass

    # may not need this? should already be defined in RegressorMixin
    # def score(self, X, y, sample_weight=None):
    #     """
    #     Returns R^2, the 'coefficient of determination' of the prediction.

    #     R^2 = 1 - u/v
    #     u = ((y_true - y_pred)**2).sum()
    #     v = ((y_true - y_true.mean())**2).sum()

    #     NOTE:
    #         Here, the X should be structures (names?), and the y are loss values

    #     Args:
    #         X: (np.arr) testing samples; shape = (n_samples, ?)
    #         y: (np.arr) true values; shape = (n_samples)
    #             NOTE: how to handle energies AND forces? does it make sense to
    #             have them be completely different entries in the database?
    #         sample_weight: (np.arr) weights for each sample in X

    #     Returns:
    #         score: (float) R^2 of  self.predict(X) w.r.t. y

    #     """
    #     pass

