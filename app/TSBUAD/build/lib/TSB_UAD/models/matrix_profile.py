import stumpy
import numpy as np

class MatrixProfile():
    """
    Wrapper of the stympy implementation of the MatrixProfile algorithm

    Parameters
    ----------
    window : int,
        target subsequence length.
    
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples - m,)
        The anomaly score.
        The higher, the more abnormal. Anomalies tend to have higher
        scores. This value is available once the detector is
        fitted.
    """

    def __init__(self, window):
        self.window = window
        self.model_name = 'MatrixProfile'

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.profile = stumpy.stump(X,m=self.window)
        #self.profile = mp.compute(X, windows=self.window)
        self.decision_scores_ = self.profile[:,0]#['mp']
        return self
    def score(self,query_length,dataset):
    
        if query_length != self.window:
            print(f"Adjusting window size: {self.window} -> {query_length}")
            adjusted_window = query_length
        else:
            adjusted_window = self.window
        
        query_profile = stumpy.stump(dataset, m=query_length)
        scores = query_profile[:, 0]  # Minimum distances for each subsequence
        return scores