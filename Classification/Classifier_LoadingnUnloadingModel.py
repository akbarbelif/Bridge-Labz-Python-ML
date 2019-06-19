import pandas as pd
import pickle


class Classifier_Serialize_Deserialize_model:

    def serializing_Model(classifier,filepath):

        classifier_model=filepath+"/classifier_model.pkl"

        #Save and Write the model to the pickle file
        with open(classifier_model,'wb') as w:
           pickle.dump(classifier,w,pickle.HIGHEST_PROTOCOL)
           #Close the pickle Model instance
           #classifier_model.close()


    def deSerializing_Model(filepath,testmodel):

        testX, testY = testmodel
        classifier_model=filepath+"/classifier_model.pkl"
        #Open and load by reading the pickle model
        with open(classifier_model,'rb') as r:
            load_model=pickle.load(r)
            accuracy=load_model.score(testX,testY)
            print("Accuracy Result for Test Data :\n",accuracy)

