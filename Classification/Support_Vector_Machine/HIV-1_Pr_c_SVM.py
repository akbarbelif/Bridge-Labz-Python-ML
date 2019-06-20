from Classification.ClassificationRegression_DataProcessing import *
from Classification.ClassificationRegression_Model import *
from Classification.Classifier_LoadingnUnloadingModel import *

class SCM_HIV:

    try:
       # Intialize SCM for HIV Class With Parameter
       def __init__(self, input_txt_path,input_path,training_path, testing_path, model_path):

           self.input_txt=input_txt_path
           self.input_data_path = input_path
           self.training_data_path = training_path
           self.testing_data_path = testing_path
           self.model_data_path = model_path
           self.outputpath = training_path, testing_path

           dataset = pd.read_csv(self.input_txt, sep=',', names=["Peptide", "Cleavage"])
           df_data=pd.DataFrame(dataset)
           # df_data.Cleavage[df_data.Cleavage == 1] = 'Yes'
           # df_data.Cleavage[df_data.Cleavage == -1] = 'No'

           df1=df_data.Peptide.apply(list)
           df_data_new=pd.DataFrame(df1.values.tolist())
           df_data=df_data.drop('Peptide', axis=1)
           df_data=pd.concat([df_data_new,df_data],axis=1)
           col_index=[0,1,2,5,6,7]
           for i in col_index:
               df_data = df_data.drop(i, axis=1)
           print(df_data)
           self.old_len=len(df_data.columns)

           print("oldLenght: ",self.old_len)
           #Count No=344 , Yes = 402
           df_data.to_csv("../Input_data/746HIV.csv", sep=',', index=False)

           # Data Processing
           ClassificationRegressionModelPreparation(self.input_data_path, self.outputpath)
           # Model Training and Testing
           self.Dec_algo()

       def Dec_algo(self):

           # import  Data for Training and Testing
           dataset = pd.read_csv(self.training_data_path)
           new_len=len(dataset.columns)
           print("New Length: ",new_len)
           y_index = dataset.columns.get_loc("Cleavage")

           X = dataset.iloc[:, self.old_len-1:new_len+1].values
           y = dataset.iloc[:, y_index:y_index + 1].values
           # print("X Axis: \n",X)
           # print("Y axis:",y)
           ##Splitting model into Training and testing
           from sklearn.model_selection import train_test_split
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

           pred_reg = ClassificationRegression_Model.SCM_classifier(X_train, y_train, X_test, y_test)
           y_pred, reg_model = pred_reg

           print("\nY_predicted Value:\n", y_pred.ravel())
           print("\nY_Ground True Value:\n", y_test.ravel())

           # Accuracy Calculation using confusion_matrix score
           Accuracy = ClassificationRegression_Model.confusion_matix(y_test, y_pred)
           print("\nAccuary Between True value and Predicted Value:\n", Accuracy)

           testing_data = X_test, y_test

           #Model Serialization (Unload)
           Classifier_Serialize_Deserialize_model.serializing_Model(reg_model, self.model_data_path)
           # model Deserialization (Load)
           Classifier_Serialize_Deserialize_model.deSerializing_Model(self.model_data_path, testing_data)
           pass

    except Exception as e:
        print(e)
        pass



model_path="../Random_forest_Classification/Model"
input_txt_path="../Input_data/746Data.txt"
input_path="../Input_data/746HIV.csv"
training_path="../Random_forest_Classification/Data/HIV_training.csv"
testing_path="../Random_forest_Classification/Data/HIV_testing.csv"
SCM_HIV(input_txt_path,input_path,training_path,testing_path,model_path)