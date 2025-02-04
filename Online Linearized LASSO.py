from sklearn import linear_model
 # we will use it for the initial batch for convenience
import scipy
import numpy as np


class Online_Linearized_Lasso_for_single_datapoint_addition(): #when the t th data point is added 
    
    #initiating the hyperparameters
    
    def init(self,learning_rate,no_of_iterations, lambda_parameter,weight,beta_tilda,total_weight,X_new,Y_new):
        
        self.learning_rate=learning_rate
        self.no_of_iterations=no_of_iterations
        self.lambda_parameter=lambda_parameter 
        self.weight=weight #a list of T weights. This is the t-independent case. Similarly we can do for t-dependent case by maling this a parameter of fit() function
        self.beta_tilda=beta_tilda
        self.total_weight=total_weight # this will depend on t
        self.X_new=X_new
        self.Y_new=Y_new
        
    #fitting the dataset to the regression model
        
    def fit(self,X_initial,Y_initial,X_new_col,Y_new_col):

    # m is the total no of data points of initial batch size

    # p is the total no of data points of the new collected batch size

    # n is the total no of dependent variables
    
        self.p=len(X_new_col)
        self.m = len(X_initial)
        self.n = len(X_initial[0])
        self.w = [0 for i in range(self.n)]
        
        self.X_initial=X_initial
        self.Y_initial=Y_initial
        self.X_new_col=X_new_col
        self.Y_new_col=Y_new_col
        
        

    #implementing gradient descent algorithm for optimisation

        for i in range(self.no_of_iterations):
            self.update_weights()

        return(self.w)    
            
        
    #function for updating the weight and bias value
        
    def update_weights(self):
        Y_prediction = [0 for i in range(self.m)]
        Y_prediction_new = [0 for i in range(self.m)]
        for i in range(self.m):
             
             Y_prediction[i] = self.predict(self.X_initial[i],self.w)
        for i in range(self.m):     
             Y_prediction_new[i] = self.predict(self.X_initial[i],self.beta_tilda)
             
         #hut   
             
        dw=[0 for i in range(self.n)]
        
        for i in range(self.n):
             c=0
             d=0
             for j in range(self.m):
                 c = c+(Y_prediction[j]-self.Y_initial[j])*self.X_initial[j][i]
                
                 d = d+(Y_prediction_new[j]-self.Y_initial[j])*self.X_initial[j][i]

             b=0
             
             for j in range(self.p):
                 b = b + (self.predict(self.X_new_col[j],self.w) - self.Y_new_col[j])*self.X_new_col[j][i]*(self.weight[j]/self.total_weight)           
                 
             if self.w[i]>0:
                 dw[i]=((c+self.lambda_parameter-d)/self.m)+b
                 
             else:
                 dw[i]=((c-self.lambda_parameter-d)/self.m)+b        
                              
             
                    
                


        for i in range(self.n):

            self.w[i] = self.w[i]-self.learning_rate*dw[i]
            

                
        
        
    #predicting the target variable
        
    def predict(self,X,w):

        s=0
        for i in range(len(w)):
            s=s+X[i]*w[i]
        return s

    def mean_squared_error(self,X,Y,w):
        s=0
        for i in range(len(X)):
            s=s + (Y[i] - self.predict(X[i],w))**2
        return (s/len(X))       

def OnlineLinearisedLASSO(X,Y,X_new,Y_new,learning_rate,no_of_iterations, lambda_parameter,weight):
    l=Online_Linearized_Lasso_for_single_datapoint_addition()
    ls=linear_model.Lasso(alpha=lambda_parameter[0]) #lamba parameter is a list with T+1 elements T = len(X_new)
    ls.fit(X,Y)
    beta_tilda=ls.coef_
    for i in range(len(Y_new)):
            
        total_weight=1
        for j in range(i):
            total_weight = total_weight + weight[j]
        l.init(learning_rate,no_of_iterations, lambda_parameter[i+1],weight,beta_tilda,total_weight,X_new[:i+1],Y_new[:i+1])
        beta_tilda=l.fit(X_initial=X,Y_initial=Y,X_new_col=X_new,Y_new_col=Y_new)
    return (beta_tilda)    
        
theta_true = 0.5*scipy.sparse.random(50, 1, density=0.04,random_state=np.random.default_rng()).A.reshape(50)
X= np.random.normal(0,1,(50,50))
Y = X.T.dot(theta_true) + np.random.normal(0,1,50)
X_new = np.random.normal(0,1,(50,50))
Y_new= X_new.T.dot(theta_true) + np.random.normal(0,1,50)
lambda_parameter=[25 for i in range(51)]
weight=[1 for i in range(50)]
beta_star=OnlineLinearisedLASSO(X=X,Y=Y,X_new=X_new,Y_new=Y_new,learning_rate=0.01,no_of_iterations=100, lambda_parameter=lambda_parameter,weight=weight)
l=Online_Linearized_Lasso_for_single_datapoint_addition()
print("mean square error for training data set=", l.mean_squared_error(X,Y,beta_star))
print("mean square error for test data set=", l.mean_squared_error(X_new,Y_new,beta_star))

    


    
            
    
    

        


