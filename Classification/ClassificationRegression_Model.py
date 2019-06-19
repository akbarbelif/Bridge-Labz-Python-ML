# @Author AKbar Belif
#Import Class Libray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from Classification.Classifier_LoadingnUnloadingModel import *

class ClassificationRegression_Model:

#Logistic_Regression
    def Logistic_reg(trainX,trainy,testX,testY):

        classifier_reg=LogisticRegression(random_state=0)
        classifier_reg.fit(trainX , trainy.ravel())
        y_pred = classifier_reg.predict(testX)

        # Visualising  the Training set Result
        from matplotlib.colors import ListedColormap
        X_Set,Y_Set=trainX,trainy.ravel()
        X1,X2=np.meshgrid(np.arange(start=X_Set[:,0].min() -1,stop=X_Set[:,0].max() +1,step=0.01),
                          np.arange(start=X_Set[:,1].min()-1,stop=X_Set[:,1].max() +1 ,step=0.01))

        plt.contourf(X1,X2,classifier_reg.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                     alpha =0.75,cmap=ListedColormap(('r','g')))
        plt.xlim(X1.min(),X2.max())
        plt.ylim(X2.min(),X2.max())
        for i,j in enumerate(np.unique(Y_Set)):
            plt.scatter(X_Set[Y_Set == j,0],X_Set[Y_Set == j, 1],c=ListedColormap(('r','g'))(i),label=j)
        plt.title("Logistic Regression for Predicting the Gender purchasing SUV (Training Set)")
        plt.xlabel("Age")
        plt.ylabel("Estimated Salary")
        plt.legend()
        plt.show()

        print("Classifiction model:/n",classifier_reg)
        print("Predication:/n",y_pred)
        log_reg_pred=[y_pred,classifier_reg]
        return log_reg_pred


#SCM_Classification
    def SCM_classifier(trainX,trainy,testX,testY):

        scmclassifier_reg=SVC(kernel='linear',random_state=0)
        scmclassifier_reg.fit(trainX ,trainy.ravel())
        y_pred = scmclassifier_reg.predict(testX)

        #Visualising  the Training set Result
        from matplotlib.colors import ListedColormap
        X_Set,Y_Set=trainX,trainy.ravel()
        X1,X2=np.meshgrid(np.arange(start=X_Set[:,0].min() -1,stop=X_Set[:,0].max() +1,step=0.01),
                          np.arange(start=X_Set[:,1].min()-1,stop=X_Set[:,1].max() +1 ,step=0.01))

        plt.contourf(X1,X2,scmclassifier_reg.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                     alpha =0.75,cmap=ListedColormap(('r','g')))
        plt.xlim(X1.min(),X2.max())
        plt.ylim(X2.min(),X2.max())
        for i,j in enumerate(np.unique(Y_Set)):
            plt.scatter(X_Set[Y_Set == j,0],X_Set[Y_Set == j, 1],c=ListedColormap(('r','g'))(i),label=j)
        plt.title("Support Vector Machine for Predicting the Gender purchasing SUV (Training Set)")
        plt.xlabel("Age")
        plt.ylabel("Estimated Salary")
        plt.legend()
        plt.show()
        scm_reg_pred = [y_pred, scmclassifier_reg]
        return scm_reg_pred


#Decision_tree_Classification
    def Decision_tree_classifier(trainX,trainy,testX,testY):

        Dec_tree_classifier_reg=DecisionTreeClassifier(criterion='entropy',random_state=0)
        Dec_tree_classifier_reg.fit(trainX , trainy.ravel())
        y_pred = Dec_tree_classifier_reg.predict(testX)

        # Visualising  the Training set Result

        from matplotlib.colors import ListedColormap
        X_Set,Y_Set=trainX,trainy.ravel()
        X1,X2=np.meshgrid(np.arange(start=X_Set[:,0].min() -1,stop=X_Set[:,0].max() +1,step=0.01),
                          np.arange(start=X_Set[:,1].min()-1,stop=X_Set[:,1].max() +1 ,step=0.01))

        plt.contourf(X1,X2,Dec_tree_classifier_reg.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                     alpha =0.75,cmap=ListedColormap(('r','g')))
        plt.xlim(X1.min(),X2.max())
        plt.ylim(X2.min(),X2.max())
        for i,j in enumerate(np.unique(Y_Set)):
            plt.scatter(X_Set[Y_Set == j,0],X_Set[Y_Set == j, 1],c=ListedColormap(('r','g'))(i),label=j)
        plt.title("Decision Tree Classification for Predicting the Gender purchasing SUV (Training Set)")
        plt.xlabel("Age")
        plt.ylabel("Estimated Salary")
        plt.legend()
        plt.show()
        dec_reg_pred = [y_pred,Dec_tree_classifier_reg]
        return dec_reg_pred

#Random Forest Classification

    def Random_forest_classifier(trainX,trainy,testX,testY):

        Rand_forest_classifier_reg=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
        Rand_forest_classifier_reg.fit(trainX , trainy.ravel())
        y_pred = Rand_forest_classifier_reg.predict(testX)

        # Visualising  the Training set Result

        from matplotlib.colors import ListedColormap
        X_Set,Y_Set=trainX,trainy.ravel()
        X1,X2=np.meshgrid(np.arange(start=X_Set[:,0].min() -1,stop=X_Set[:,0].max() +1,step=0.01),
                          np.arange(start=X_Set[:,1].min()-1,stop=X_Set[:,1].max() +1 ,step=0.01))

        plt.contourf(X1,X2,Rand_forest_classifier_reg.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                     alpha =0.75,cmap=ListedColormap(('r','g')))
        plt.xlim(X1.min(),X2.max())
        plt.ylim(X2.min(),X2.max())
        for i,j in enumerate(np.unique(Y_Set)):
            plt.scatter(X_Set[Y_Set == j,0],X_Set[Y_Set == j, 1],c=ListedColormap(('r','g'))(i),label=j)
        plt.title("Random Forest Classification for Predicting the Gender purchasing SUV (Training Set)")
        plt.xlabel("Age")
        plt.ylabel("Estimated Salary")
        plt.legend()
        plt.show()
        rand_reg_pred=[y_pred,Rand_forest_classifier_reg]
        return rand_reg_pred

#Calculate R2 Score for Accuracy Measurement
    @staticmethod
    def r2_cal(y_true,y_pred):

        accuracy=r2_score(y_true,y_pred)
        return accuracy


#Calculate Predication for Accurate result
#Using confusion_matrix which gives both  Correct and Incorrect output

    @staticmethod
    def confusion_matix(y_true,y_pred):

        from sklearn.metrics import confusion_matrix
        accuracy=confusion_matrix(y_true,y_pred)
        return accuracy
