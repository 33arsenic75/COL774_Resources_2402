import cvxopt
import numpy as np
from scipy.spatial.distance import cdist

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):
        self.X : np.array = None
        self.y : np.array = None
        self.kernel : str = None
        self.C : float = None
        self.gamma : float = None
        self.P : np.array = None
        self.q : np.array = None
        self.c : float = None
        self.alphas : np.array = None 
        self.support_vectors : np.array = None 
        self.support_vector_labels : np.array = None
        self.support_vector_indices : np.array = None
        self.eps : float = 1e-4
        pass
        
    def fit(self, X, y, kernel = 'linear', C = 1.0, gamma = 0.001, dump = False):
        '''
        Learn the parameters from the given training data
        Classes are 0 or 1
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            kernel: str
                The kernel to be used. Can be 'linear' or 'gaussian'
                
            C: float
                The regularization parameter
                
            gamma: float
                The gamma parameter for gaussian kernel, ignored for linear kernel
        '''
        self.X = X
        self.y = y
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

        N, D = X.shape
        if kernel == 'linear':
            K = X @ X.T
        elif kernel == 'gaussian':
            # K = np.zeros((N, N))
            # for i in range(N):
            #     for j in range(N):
            #         K[i,j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j]) ** 2)
            pairwise_sq_dists = cdist(X, X, metric='sqeuclidean')  # Compute squared Euclidean distances efficiently
            K = np.exp(-gamma * pairwise_sq_dists)
        else:
            raise ValueError("Unsupported kernel type")
        
        y_diag = np.diag(self.y.astype(float))
        P = cvxopt.matrix(y_diag @ K @ y_diag, tc = 'd')
        q = cvxopt.matrix(-np.ones(N))
        G = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
        A = cvxopt.matrix(self.y.astype(float), (1, N))
        b = cvxopt.matrix(0.0)
        if dump:
            print(f"P: {P.size}\n {P}")    
            print(f"q: {q.size}\n {q}")    
            print(f"G: {G.size}\n {G}")
            print(f"h: {h.size}\n {h}")
            print(f"A: {A.size}\n {A}")
            print(f"b: {b.size}\n {b}")

        # Solve the quadratic optimization problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P = P, q =  q, G = G, h = h, A = A, b = b) 
        alphas = np.ravel(solution['x'])
        # # Extract support vectors
        self.support_vector_indices = alphas > self.eps
        self.alphas = alphas[self.support_vector_indices]
        self.support_vectors = X[self.support_vector_indices]
        self.support_vector_labels = self.y[self.support_vector_indices]
        
        # Compute weight vector and bias for linear kernel
        if kernel == 'linear':
            self.w = np.sum(self.alphas[:, None] * self.support_vector_labels[:, None] * self.support_vectors, axis=0)
            self.b = np.mean(self.support_vector_labels - self.support_vectors @ self.w)
        else:
            self.w = None
            self.b = np.mean(self.support_vector_labels - np.sum(self.alphas * self.support_vector_labels * K[self.support_vector_indices][:, self.support_vector_indices], axis=1))
    
    def predict(self, X):
        '''
        Predict the class of the input data
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)
        '''
        if self.kernel == 'linear':
            # Decision function using support vectors
            decision_values = np.sum((self.alphas * self.support_vector_labels)[:, np.newaxis] * (self.support_vectors @ X.T), axis=0)
        else:
            # Compute the kernel similarity between test points and support vectors
            # K = np.exp(-self.gamma * np.linalg.norm(X[:, np.newaxis] - self.support_vectors, axis=2) ** 2)
            sq_dists = cdist(X, self.support_vectors, metric='sqeuclidean')
            K = np.exp(-self.gamma * sq_dists)
            decision_values = np.sum(self.alphas * self.support_vector_labels * K, axis=1)
        
        return np.sign(decision_values + self.b)  # Output is -1 or 1

        