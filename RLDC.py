# -*- coding: utf-8 -*-
"""
RLDC algorithm. Use the five - fold cross - validation method to obtain the accuracy metric of classification performance,Run it 10 times.
"""
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance

#Function for calculating classification performance evaluation metrics*********************************************************************

def g_mean_score(y_true, y_pred):   
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = np.diag(cm) / np.sum(cm, axis=1)
    g_mean = np.sqrt(np.prod(sensitivities))
    return g_mean

 # Metric calculation
def calculate_metrics(y, y_predict,Jresult):
     #print("the result of sklearn package")
     auc = roc_auc_score(y, y_predict)
     #print("sklearn auc:",auc)
     g_mean= g_mean_score(y, y_predict)
     #print("sklearn accuracy:",accuracy)
     recal = recall_score(y, y_predict)
     precision = precision_score(y, y_predict)
     F1_sc=(2*recal*precision)/(recal+precision)
     new_r=[auc,g_mean,F1_sc]
     Jresult.extend(new_r)
    
 #GNB Classification Performance Evaluation Function
def fenlei(x_train,y_train,x_test,y_test):
   Jresult=[]
   from sklearn.naive_bayes import GaussianNB
   gnb = GaussianNB()   
   gnb.fit(x_train,y_train) 
   calculate_metrics(y_test,gnb.predict(x_test),Jresult) 
   return Jresult


# Main function*************************************************************************************
if __name__ == '__main__':
    
    # Create an empty DataFrame to store experiment results
    experiment_results = pd.DataFrame()
    column_names = ['GNB_AUC','GNB_Gmean','GNB_F1']
    # Create empty columns with column names and add them to the DataFrame
    for column_name in column_names:
        experiment_results[column_name] = []
        
    # Set experimental parameters, including the name of the dataset and the number  of division intercepts.*******************************
    datasetY = pd.read_csv('iris.csv')
    t=9
    #**********************************************************************
    
    # Split the dataset
    global X, y
    y = datasetY['class']
    X = datasetY.drop(columns='class')
    # Calculate the distance matrix between all points
    A = distance.cdist(X, X, 'euclidean')
    # Calculate the maximum distance among all points.
    b =np.max(A[A > 0])
    # Create a five-fold stratified cross-validation object
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Repeat the experiment 10 times
    for i in range(10):
        # Perform five-fold cross-validation

            for train_index, test_index in kfold.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                # Initialize a DataFrame to store the proportions under different intercepts
                x_train1 = pd.DataFrame(index=X_train.index)
                x_test1 = pd.DataFrame(index=X_test.index)

                # Calculate the proportion of minority class points under different intercepts.
                for factor in np.arange(1,t):
                    current_b = factor * b/t
                    minority_indices = np.where(y_train == 0)[0] 
                    for i in X_train.index:
                        within_distance_indices = np.where(A[i, train_index] <= current_b)[0]
                        within_minority_count = np.sum(np.isin(within_distance_indices, minority_indices))
                        total_within_count = len(within_distance_indices-1)
                        x_train1.loc[i, f'{factor}b'] = within_minority_count / total_within_count if total_within_count > 0 else 0.0

                    for i in X_test.index:
                        within_distance_indices = np.where(A[i, train_index] <= current_b)[0]
                        within_minority_count = np.sum(np.isin(within_distance_indices, minority_indices))
                        total_within_count = len(within_distance_indices)
                        x_test1.loc[i, f'{factor}b'] = within_minority_count / total_within_count if total_within_count > 0 else 0.0
                
                Danswer =fenlei(x_train1, y_train, x_test1, y_test)
                Danswer= np.array(Danswer).reshape(1,3)
                Dw=pd.DataFrame(Danswer,columns=experiment_results.columns)
                experiment_results=pd.concat([experiment_results,Dw])
  
    fw1 = experiment_results.mean(axis=0)         
    # Print the average of the final results
    print(fw1)
    
    
 
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
