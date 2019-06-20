#Import Class Library

from Classification.ClassificationRegression_DataProcessing import *
from Classification.ClassificationRegression_Model import *
from Classification.Classifier_LoadingnUnloadingModel import *

class Rand_forest_classifier:

    try:
       # Intialize SCM Class With Parameter
       def __init__(self, input_path, training_path, testing_path, model_path):

           self.input_data_path = input_path
           self.training_data_path = training_path
           self.testing_data_path = testing_path
           self.model_data_path = model_path
           self.outputpath = training_path, testing_path
           # Data Processing
           ClassificationRegressionModelPreparation(self.input_data_path, self.outputpath)
           # Model Training and Testing
           self.rand_algo()

       def rand_algo(self):

           # import  Data for Training and Testing
           dataset = pd.read_csv(self.training_data_path)
           print(dataset.head())
           X_index1 = dataset.columns.get_loc("0")
           X_index2 = dataset.columns.get_loc("1")
           y_index = dataset.columns.get_loc("Purchased")

           X = dataset.iloc[:, [X_index1, X_index2]].values
           y = dataset.iloc[:, y_index:y_index + 1].values

           ##Splitting model into Training and testing
           from sklearn.model_selection import train_test_split
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

           pred_reg = ClassificationRegression_Model.Random_forest_classifier(X_train, y_train, X_test, y_test)
           y_pred, reg_model = pred_reg

           print("\nY_predicted Value:\n", y_pred.ravel())
           print("\nY_Ground True Value:\n", y_test.ravel())

           # Accuracy Calculation using confusion_matrix score
           Accuracy = ClassificationRegression_Model.confusion_matix(y_test, y_pred)
           print("\nAccuary Between True value and Predicted Value:\n", Accuracy)

           testing_data = X_test, y_test

           ##Model Serialization (Unload)
           Classifier_Serialize_Deserialize_model.serializing_Model(reg_model, self.model_data_path)
           # model Deserialization (Load)
           Classifier_Serialize_Deserialize_model.deSerializing_Model(self.model_data_path, testing_data)
           pass

    except Exception as e:
        print(e)
        pass



model_path="../Random_forest_Classification/Model"
input_path="../Input_data/Social_Network_Ads.csv"
training_path="../Random_forest_Classification/Data/training.csv"
testing_path="../Random_forest_Classification/Data/testing.csv"
Rand_forest_classifier(input_path,training_path,testing_path,model_path)

