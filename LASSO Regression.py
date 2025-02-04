#creating a class for LASSO Regression
# Python has a LASSO Regression package. I have implemented the source code for LASSO Regression so that
# I can use this as a guideline for coding online linearized LASSO

class Lasso_Regression():
    
    #initiating the hyperparameters
    
    def init(self,learning_rate,no_of_iterations, lambda_parameter):
        
        self.learning_rate=learning_rate
        self.no_of_iterations=no_of_iterations
        self.lambda_parameter=lambda_parameter
        
    #fitting the dataset to the regression model
        
    def fit(self,X,Y):

    # m is the total no of data points

    # n is the total no of dependent variables
    
        self.m = len(X)
        self.n = len(X[0])
        self.w = [0 for i in range(self.n)]
        self.b = 0
        self.X=X
        self.Y=Y

    #implementing gradient descent algorithm for optimisation

        for i in range(self.no_of_iterations):
            self.update_weights()

            
        
    #function for updating the weight and bias value
        
    def update_weights(self):
        Y_prediction = [0 for i in range(self.m)]
        for i in range(self.m):
             
             Y_prediction[i] = self.predict(self.X[i])
             
        dw=[0 for i in range(self.n)]

        d=0
        for j in range(self.m):
            
             d=d+2*(Y_prediction[j]-self.Y[j])
        db = d/self.m
        
        for i in range(self.n):
             c=0
             for j in range(self.m):
                 c=c+2*(Y_prediction[j]-self.Y[j])*self.X[j][i]
                 
             if self.w[i]>0:
                 dw[i]=(c+self.lambda_parameter)/self.m
                 
             else:
                 dw[i]=(c-self.lambda_parameter)/self.m        
                              
             
                    
                


        for i in range(self.n):

            self.w[i] = self.w[i]-self.learning_rate*dw[i]
            
        self.b = self.b - self.learning_rate*db

                
        
        
    #predicting the target variable
        
    def predict(self,X):ftg

        s=0
        for i in range(len(X)):
            s=s+X[i]*self.w[i]
        return s+self.b

    def mean_squared_error(self):
        s=0
        for i in range(self.m):
            s=s + (self.Y[i] - self.predict(self.X[i]))**2
        print("mean squared error = ",s/self.m)            

        
#main
l = Lasso_Regression()

for 
l.init(learning_rate=0.001,no_of_iterations=20, lambda_parameter=0) #these values can be optimised using cross validation
X=[[4,5],[6,7],[8,9]] # Real values  w = [1,1] b=0
Y=[9,13,17]
l.fit(X,Y)
print("weights=",l.w)
print("bias=",l.b)
# the values that we got are close enough to the real values but not overfitted
l.mean_squared_error()


