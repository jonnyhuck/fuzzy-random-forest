"""
Fuzzy Random Forest Classifier (class-based)
"""
import numpy as np
from scipy.stats import beta
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class FuzzyRF:
    """
    Class for Fuzzy Random Forest classifier with Beta simulators.
    """

    def __init__(self, input_data, trees=20, branches=8):
        """
        Constructor
        """
        # instance variables
        self.trees = trees
        self.branches = branches
        self.rng = np.random.default_rng()
        self.rows, self.cols = input_data[0].shape[:2]

        # load raster and vector training data
        self.stacked_array, self.X, self.y = input_data

        # separate data into train and test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # train standard random forest model
        self.clf = RandomForestClassifier(n_estimators=self.trees, max_depth=self.branches, random_state=42)
        self.clf.fit(X_train, y_train)

        # calculate confusion matrix
        y_pred = self.clf.predict(X_test)
        self.cm = confusion_matrix(y_test, y_pred, normalize='true')

        # compute tree probabilities for all pixels
        n_samples = self.stacked_array.shape[0] * self.stacked_array.shape[1]
        X_img = self.stacked_array.reshape(n_samples, self.stacked_array.shape[2])
        self.tree_probs = np.array([ est.predict_proba(X_img) for est in self.clf.estimators_ ])  # (trees, cells, classes)
        self.classes = self.tree_probs.shape[2]

        # combine probabilities with those from the confusion matrix using power posterior
        self._adjust_tree_probs()

        # compute Beta parameters (method-of-moments)
        class_probs = self.tree_probs.transpose(2,1,0)  # (classes, cells, trees)
        self.a, self.b = self.method_of_moments(class_probs)


    def _adjust_tree_probs(self):
        """
        Use Power Posterior to update the tree probabilities to account for the confusion matrix 
        """
        log2 = np.log(2)
        for t in range(self.tree_probs.shape[0]):
            P = self.tree_probs[t]
            ec = np.argmax(P, axis=1)
            Q = self.cm[ec]

            # Jensen-Shannon divergence
            js = self.js_divergence(P, Q)
            alpha = np.clip(1.0 - js / log2, 0.0, 1.0)

            # power posterior
            tempered = Q ** alpha[:, None]
            P *= tempered

            # handle zero-sum rows
            row_sums = P.sum(axis=1, keepdims=True)
            zero_mask = row_sums[:, 0] == 0
            P[zero_mask, :] = 1.0 / P.shape[1]

            # normalize
            P /= P.sum(axis=1, keepdims=True)
            self.tree_probs[t] = P


    def kl_divergence(self, P, Q, eps=1e-12):
        """
        Kullback Liebler divergence
        P, Q: arrays of shape (cells, classes)
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        """
        P = np.clip(P, eps, 1)
        Q = np.clip(Q, eps, 1)
        return np.sum(P * np.log(P / Q), axis=1)


    def js_divergence(self, P, Q, eps=1e-12):
        """
        Jensen–Shannon divergence per row
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        """
        M = 0.5 * (P + Q)
        return 0.5 * self.kl_divergence(P, M, eps) + 0.5 * self.kl_divergence(Q, M, eps)


    def method_of_moments(self, class_probs):
        """
        Fast method to estimate beta parameters
        https://en.wikipedia.org/wiki/Method_of_moments_(statistics)
        """
        # get mean and variance of values
        mean = np.mean(class_probs, axis=2)
        var = np.var(class_probs, axis=2, ddof=1)

        # ensure valid numbers
        eps = 1e-6
        mean_safe = np.clip(mean, eps, 1-eps)
        max_var = mean_safe * (1-mean_safe) - eps
        var_safe = np.clip(var, eps, max_var)

        # compute Beta params
        common_factor = np.maximum(mean_safe * (1-mean_safe)/var_safe - 1, eps)
        a = mean_safe * common_factor
        b = (1 - mean_safe) * common_factor

        # handle edge cases
        zeros_mask = mean <= eps
        ones_mask = mean >= 1 - eps
        a[zeros_mask] = 1.0
        b[zeros_mask] = 1e6
        a[ones_mask] = 1e6
        b[ones_mask] = 1.0

        # final safety check and return
        a = np.clip(a, eps, None)
        b = np.clip(b, eps, None)
        return a, b


    def get_params(self):
        """ Return the beta parameters"""
        return self.a, self.b
    

    def mc_draws(self, n_draws):
        """ 
        Draw a number of new fuzzy rasters for Monte Carlo analysis using CRN (one randon mumber per landscape) 
        
        Non CRN version:
            mc_draws = self.rng.beta(self.a[:,:,None], self.b[:,:,None], size=(self.a.shape[0], self.a.shape[1], n_draws))
        """
        # Draw common uniforms (one per draw) and reshape for broadcasting
        x = self.rng.uniform(0.0, 1.0, size=n_draws)  # (n_draws,)
        x = x[None, None, :]  # (1, 1, n_draws)

        # use the randon mumbers to draw n random landscapes
        mc = beta.ppf(x, self.a[:, :, None], self.b[:, :, None])

        # normalize across classes
        mc /= mc.sum(axis=0, keepdims=True)

        # reshape into landscapes and return
        return mc.transpose(2, 0, 1).reshape(n_draws, self.classes, self.rows, self.cols)   # (draws, classes, rows, cols)


# example usage
if __name__ == "__main__":

    from geopandas import read_file
    from rasterio import open as rio_open
    from rasterio.features import rasterize
    from matplotlib.pyplot import imshow, show


    def prepare_training_data(raster_path, bands_to_use, vector_path):
        """
        Prep the training data
        """
        # load raster dataset
        with rio_open(raster_path) as src:
            
            # extract required bands and shift axes
            bands = src.read(bands_to_use)
            stacked_array = np.moveaxis(bands, 0, -1)  # (rows, cols, bands)

            # load vector training dataset, ensure class data is numeric
            training_data = read_file(vector_path).to_crs("EPSG:32630")
            training_data['_mode'] = training_data['_mode'].astype(int)
            
            # THIS IS JUST FOR MY DEMO DATA!!!
            training_data = training_data.sample(frac=0.2)  

            # rasterize training geometries and mask out of imagery
            label_raster = rasterize( zip(training_data.geometry, training_data['_mode']), 
                                     out_shape=(src.height, src.width), transform=src.transform)
            mask = label_raster > 0
            X = stacked_array[mask]
            y = label_raster[mask]

        # return 
        return stacked_array, X, y


    RASTER_PATH = '../data/Arnside_Silverdale_no_crop.tif'
    BANDS = [2, 3, 4, 8]
    VECTOR_PATH = "../data/Download_Silverdale_2532092/lcm-2021-vec_5552404/lcm-2021-vec_5552404.gpkg"
    
    # prepare training data
    print('prep...')
    prep = prepare_training_data(RASTER_PATH, BANDS, VECTOR_PATH)

    # create an instance of the fuzzy random forest class
    print('training...')
    frf = FuzzyRF(prep, trees=20, branches=3)
    
    # use it to draw n landscapes
    print('drawing...')
    landscapes = frf.mc_draws(5)
    print(f"Returns {landscapes.shape[0]} draws, {landscapes.shape[1]} classes, for a landscape of {landscapes.shape[3]}x{landscapes.shape[2]} cells.")

    # get the median of the draws
    medians = np.median(landscapes, axis=0)
    
    # show the output surface for each class
    imshow(medians[0])
    show()