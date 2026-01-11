import numpy as np
import pandas as pd


class LinearRegression:
    
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.weight=None 
        self.bias=None   
        #track loss 
        self.train_loss=[]
        self.val_loss=[]
        
    #training out dataset
    def fit(self,x_train,y_train,x_val=None,y_val=None):
        n_samples,n_features =x_train.shape
        self.weight =np.zeros(n_features) #initialising weight to 0 for all features
        self.bias =0 #initialising bias as 0 (there is only one bias)
        
        #now predicting values as y_pred=weights*x +bias
        
        for _ in range(self.n_iter):
           #training the model
           y_pred=np.dot(x_train,self.weight) + self.bias
        
           #now calculating gradient of weights and bias
           dw=(1/n_samples)*np.dot(x_train.T,(y_pred-y_train))
           db=(1/n_samples)*np.sum(y_pred-y_train) 
        
           #now updates the weights and bias 
           self.weight=self.weight-(self.lr*dw)
           self.bias=self.bias-(self.lr*db)
           
           #--------Train loss----------
           train_loss=np.mean((y_train-y_pred)**2)
           self.train_loss.append(train_loss)
           
           
           #--------Validation loss---------
           if x_val is not None and y_val is not None:
               y_val_pred=np.dot(x_val,self.weight) +self.bias
               val_loss=np.mean((y_val_pred-y_val)**2)
               self.val_loss.append(val_loss)
               
           
        
    def prediction(self,x):
        y_pred=np.dot(x,self.weight) + self.bias
        return y_pred
        
        
        