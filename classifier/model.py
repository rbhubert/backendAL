# All kind of imports...

import fasttext

from classifier.preprocessing import preprocessing_google, preprocessing_reddit, preprocessing_twitter
from db.database import modelDB
from enums import sources_base, model

directory_files = "./files/"
directory_models = "./models/"


class DeepLearningModel:
    def __init__(self, name, newModel, dataType=""):
        self.original_name = name
        self.model_id = name.lower().replace(" ", "_")

        if newModel:
            self.model = None
            self.dataType = dataType
            self.save_model(only_db=True)
        else:
            model_info = modelDB.get_model(self.model_id)
            self.dataType = model_info[model.DATA_TYPE]

            if model.FILE in model_info:
                self.model = fasttext.load_model(model_info[model.FILE])

        if self.dataType == model.ModelDataType.GOOGLE:
            self.preprocess = preprocessing_google
        elif self.dataType == model.ModelDataType.REDDIT:
            self.preprocess = preprocessing_reddit
        else:
            self.preprocess = preprocessing_twitter

    def __to_file(self, training_set, training=True):
        training_set[sources_base.TEXT] = training_set[sources_base.TEXT].apply(lambda x: self.preprocess(x))

        train_test = "train" if training else "test"
        file_train = directory_files + train_test + "___" + self.model_id + ".txt"

        # prepare data

        file = open(file_train, "w")
        for index, row in training_set.iterrows():
            print("ROW", row)
            print(row[sources_base.TEXT])
            print(row[sources_base.CLASSIFICATION])
            print(self.original_name)
            print(row[sources_base.CLASSIFICATION][self.original_name])
            line = "__label__" + row[sources_base.CLASSIFICATION][self.original_name] + ' ' + row[sources_base.TEXT]
            file.write(line + "\n")

        return file_train

    def train(self, training_set):
        file_train = self.__to_file(training_set)
        self.model = fasttext.train_supervised(file_train, epoch=50)
        self.save_model()
        return True

    def save_model(self, only_db=False):
        if only_db:
            model_info = {
                model.ID: self.model_id,
                model.ORIGINAL_NAME: self.original_name,
                model.DATA_TYPE: self.dataType
            }
            modelDB.save_model(model_info)
            return True

        model_file = directory_models + self.model_id + ".bin"
        self.model.save_model(model_file)
        model_info = {
            model.ID: self.model_id,
            model.ORIGINAL_NAME: self.original_name,
            model.FILE: model_file,
            model.DATA_TYPE: self.dataType
        }
        modelDB.save_model(model_info)
        return True

    def predict(self, document_text):
        text = self.preprocess(document_text)
        result = self.model.predict(text)
        to_return = {sources_base.CLASSIFICATION_MODEL: result[0][0],
                     sources_base.CLASSIFICATION_PROBABILITY: result[1][0]}

        return to_return

    def performance(self, testing_set):
        file_test = self.__to_file(testing_set, training=False)
        result = self.model.test(file_test)
        return result
