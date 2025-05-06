import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA).

    Reduction of dimension of dataset using either Eigen Value Decomposition (EVD)
    or Singular Value Decomposition (SVD). 

    Parameters
    ----------

    n_components: int, default = None
        Number of componnts to keep, if None all components are kept.

    method: str, default = "svd"
        if method == "svd": solved by Singular Value Decomposition.
        if method == "evd": solved by Eigen Value Decomposition.

    Attributes
    ----------
    eigen_vectors: ndarray
        Principal axes representing the directions of maximum variance.

    """
    def __init__(self, n_components=None, method="svd"):
        self.n_components = n_components
        self.method = method
        self.eigen_vectors = None

    def fit(self, X):
        """
        Fit the model with X.

        Parameters
        ----------

        X: Training Data.

        """

        # Centering the data
        X = X - np.mean(X, axis=0)

        if self.method == "svd":

            # Singular value decomposition
            U, S, self.VT = np.linalg.svd(X, full_matrices=False)

            self.eigen_vectors = self.VT

        elif self.method == "evd":

            cov_matrix = np.cov(X.T)

            # Eigen value decomposition
            eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

            # Sorting the vectors with maximum variance (eigenvalues)
            sorted_index = np.argsort(eig_vals)[::-1]
            eig_vals = eig_vals[sorted_index]
            self.eig_vecs = eig_vecs[:, sorted_index]

            self.eigen_vectors = self.eig_vecs

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.

        Parameters
        ----------

        X: Array on which dimensionality reduction is applied.

        Returns
        -------

        Returns the transformed X. 
        """
        if self.method == "svd":
                
            # Transforming X
            X_svd_proj = X @ self.VT.T[:, : self.n_components]
                
            return X_svd_proj
            
        elif self.method == "evd":
                
            # Transforming X
            X_evd_proj = X @ self.eig_vecs[:, : self.n_components]
                
            return X_evd_proj