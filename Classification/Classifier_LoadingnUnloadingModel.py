import pandas as pd
import pickle


class Classifier_Serialize_Deserialize_model:

    def serializing_Model(classifier,filepath):

        classifier_model=filepath+"/classifier_model.pkl"
        print(classifier_model)
        # Save and Write the model to the pickle file
        # with open(self.classifier_model,'wb') as w:
        #    pickle.dump(self.classifier_model,w,pickle.HIGHEST_PROTOCOL)
        #    #Close the pickle Model instance
        #    classifier_model.close()


    def deSerializing_Model(self,classifier,filepath,testmodel):

        testX, testY = self.testmodel
        #Open and load by reading the pickle model
        with open(self.classifier_model,'rb') as r:
            load_model=pickle.load(r)
            accuracy=load_model.score(testX,testY)
            print("Accuracy Result for Test Data :\n",accuracy)

