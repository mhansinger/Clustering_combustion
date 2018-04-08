import numpy as np
import pandas as pd

class lpca:

    def __init__(self, X, n_components=None, regions=1):
        """
        PCA is a form of dimensionality reduction that relies
        on computing the covariance matrix, then redrawing the
        axes along the vectors of highest correlation. To do that
        we'll use the eigenvecors and eigenvalues of the covariance
        matrix. We'll draw as many axes as n_components.
        ---
        KWargs:
        n_components: how many components to have after dim.
        reduction (int)
        """
        self.n_components = n_components
        self.regions = regions
        self.X = None
        self.lpca_container = None

    def fit(self,X):
        """
        After getting the covariance matrix, calculates
        the eigenvectors and values. The eigenvectors lie
        along the line of highest correlation, and we'll use
        the strongest "n_components" number of axes. The
        strength is determined by the eigenvalues, which
        are also used to measure the explained variance.
        ---
        Input: X, data matrix (dataframe, array, list of lists)
        """

        self.X = self.convert_to_array(X)

        self.covariance_matrix = self.get_covariance_matrix(self.X)
        self.get_eigenvalues_and_eigenvectors(self.covariance_matrix)

        indices = np.argsort(self._eigenvalues)  # smallest to largest
        indices = np.flip(indices, axis=0)  # largest to smallest
        self.ranked_eigenvalues = self._eigenvalues[indices]
        self.ranked_eigenvectors = self._eigenvectors[:, indices]
        self.explained_variance = self.ranked_eigenvalues / np.sum(self.ranked_eigenvalues)

        if self.n_components:
            self.ranked_eigenvalues = self.ranked_eigenvalues[:self.n_components]
            self.ranked_eigenvectors = self.ranked_eigenvectors[:self.n_components]
            self.explained_variance = self.explained_variance[:self.n_components]

    def get_covariance_matrix(self, X):
        """
        Computes the covariance matrix by removing the means and
        multiplying the transpose of the data matrix with itself.
        """
        column_averages = np.mean(X, axis=0)
        demeaned_matrix = X - column_averages
        return np.dot(demeaned_matrix.T, demeaned_matrix) / (X.shape[0] - 1)

    def get_eigenvalues_and_eigenvectors(self, cov_mat):
        """
        Use the built in eigen extractor from numpy.
        Based on LAPACK eigen solver.
        """
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(cov_mat)

    def transform(self, X):
        """
        Converts the new data into the lower dimensional
        space. This is done by projecting the data along
        the eigenvectors that were created when fitting
        to the original data.
        ---
        Input: X, data matrix (dataframe, array, list of lists)
        """
        X = self.convert_to_array(X)
        column_averages = np.mean(X, axis=0)
        demeaned_matrix = X - column_averages
        return np.dot(demeaned_matrix, self.ranked_eigenvectors.T)

    def fit_transform(self, X):
        """
        Fits and Transforms the data returning the
        representation of the training data in the
        lower dimensional space.
        """
        self.fit(X)
        return self.transform(X)

    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1, 2])):
            return x
        return np.array(x)

    def handle_1d_data(self, x):
        """
        Converts 1 dimensional data into a series of rows with 1 columns
        instead of 1 row with many columns.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    def convert_to_array(self, x):
        """
        Takes in an input and converts it to a numpy array
        and then checks if it needs to be reshaped for us
        to use it properly
        """
        x = self.pandas_to_numpy(x)
        x = self.handle_1d_data(x)
        return x

    def get_lpca(self,X):
        '''
        compute the local pca, components for each region
        '''

        self.lpca_container = np.zeros((self.X.shape[0],self.X.shape[1],self.regions))



