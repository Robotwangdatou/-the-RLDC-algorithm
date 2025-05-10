# -*- coding: utf-8 -*-
"""
Optimized RLDC algorithm code. Uses five-fold cross-validation method to obtain classification performance metrics, run 10 times.
"""
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)
import pandas as pd

# ======================= Core Algorithm Module =======================
def rldc_feature_generation(X, y, thresholds):
    """Core algorithm for RLDC feature generation"""
    # Calculate global maximum distance of training set
    tree = KDTree(X)
    max_dist = np.max(tree.query(X, k=2)[0][:, 1])  # Maximum distance of nearest neighbors
    
    n_samples = X.shape[0]
    features = np.zeros((n_samples, thresholds))
    minority_mask = (y == 1)  # Assuming minority class label is 1
    
    for i in range(n_samples):
        # Batch query neighborhoods for all radii
        neighbors_indices, neighbors_distances = tree.query_radius(
            [X[i]], 
            r=max_dist,
            return_distance=True
        )
        neighbors_indices = neighbors_indices[0]
        distances = neighbors_distances[0]
        
        # Precompute all threshold conditions
        for t in range(thresholds):
            current_r = (t+1) * max_dist / thresholds
            within_radius = (distances <= current_r)
            
            n_minor = np.sum(minority_mask[neighbors_indices[within_radius]])
            n_major = np.sum(~minority_mask[neighbors_indices[within_radius]])
            
            features[i, t] = n_minor / (n_major + 1e-8)  # Smoothing handling
    
    return features

def gmean_score(y_true, y_pred):
    """Optimized geometric mean calculation"""
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = np.diag(cm) / (cm.sum(axis=1) + 1e-8)  # Avoid division by zero
    return np.sqrt(np.prod(sensitivities))

# ======================= Experimental Framework Module =======================
def process_fold(X_train, X_test, y_train, y_test, thresholds):
    """Single fold feature generation and evaluation"""
    # Training set feature generation
    train_features = rldc_feature_generation(X_train.values, y_train.values, thresholds)
    
    # Test set feature generation (Using training set's KDTree structure)
    test_features = np.zeros((len(X_test), thresholds))
    train_tree = KDTree(X_train.values)
    max_dist = np.max(train_tree.query(X_train.values, k=2)[0][:, 1])
    minority_mask = (y_train == 1).values
    
    for i, (index, row) in enumerate(X_test.iterrows()):
        neighbors_indices, distances = train_tree.query_radius(
            [row.values], 
            r=max_dist,
            return_distance=True
        )
        neighbors_indices = neighbors_indices[0]
        distances = distances[0]
        
        for t in range(thresholds-1):
            current_r = (t+1) * max_dist / thresholds
            within_radius = (distances <= current_r)
            
            n_minor = np.sum(minority_mask[neighbors_indices[within_radius]])
            n_major = np.sum(~minority_mask[neighbors_indices[within_radius]])
            
            test_features[i, t] = n_minor / (n_major + 1e-8)
    
    # Model training and evaluation
    model = GaussianNB()
    model.fit(train_features, y_train)
    probs = model.predict_proba(test_features)[:, 1]
    y_pred = (probs >= 0.5).astype(int)  # Classification threshold
    
    return {
        'AUC': roc_auc_score(y_test, probs),
        'F1': f1_score(y_test, y_pred),
        'G-Mean': gmean_score(y_test, y_pred)
        
    }

# ======================= Main Execution Flow =======================
if __name__ == '__main__':
    
    # Number of experiments
    num_experiments = 10
    results = []
    
    for _ in range(num_experiments):
        # Data preparation
        dataset = pd.read_csv('iris.csv')
        X = dataset.drop(columns='class')
        # Convert class labels to numerical
        y = dataset['class'].astype('category').cat.codes
    
        # Experiment parameters
        thresholds = 9
    
        # Cross - validation process
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
            # Calculate evaluation metrics for the current fold
            metrics = process_fold(X_train, X_test, y_train, y_test, thresholds)
            results.append(metrics)
    
    # Aggregate the results
    results_df = pd.DataFrame(results)
    print("\nExperimental Results Summary:")
    print(results_df.mean().round(4))