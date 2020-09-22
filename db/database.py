import os

import pymongo

from enums import model, sources_base
from enums import news, tweets, reddit
from enums.config import *
from enums.database import DBCollections
from utils.singleton import Singleton

MONGODB_URI = os.environ.get('MONGODB_URI', DATABASE)
MONGODB_NAME = os.environ.get('MONGODB_NAME', DATABASE_NAME)


# Database connection. This class is a Singleton.
class Database(metaclass=Singleton):
    def __init__(self):
        client = pymongo.MongoClient(MONGODB_URI)
        db_name = MONGODB_NAME
        self.__db = client[db_name]

    def __getitem__(self, item):
        return self.__db[item]


class RecoveredInfo:
    def __init__(self, collection_type):
        self.db = Database()
        self.collection = collection_type

    def add_info(self, info_item):
        id_dict = self.__get_id()
        return self.db[self.collection].replace_one({id_dict: info_item[id_dict]}, info_item, upsert=True)

    def add_classification_user(self, info_id, classification):
        id_dict = self.__get_id()
        return self.db[self.collection].update_one({id_dict: info_id},
                                                   {"$set": {sources_base.CLASSIFICATION: classification}})

    def add_classification_model(self, info_id, model_name, classification):
        id_dict = self.__get_id()
        path = sources_base.CLASSIFICATION_BY_MODEL + "." + model_name
        return self.db[self.collection].update_one({id_dict: info_id},
                                                   {"$set": {path: classification}})

    def __get_id(self):
        if self.collection == DBCollections.NEWS:
            return news.URL
        elif self.collection == DBCollections.TWITTER:
            return tweets.TWEET_ID
        else:
            return reddit.PERMANENT_LINK

    def get_info(self, info_id):
        id_dict = self.__get_id()
        return self.db[self.collection].find_one({id_dict: info_id}, {"_id": 0})

    def exist_info(self, info_id):
        id_dict = self.__get_id()
        return self.db[self.collection].find_one({id_dict: info_id}, {"_id": 0}) is not None

    def get_all_infoItems(self):
        return self.db[self.collection].find({}, {"_id": 0})

    def get_all_by_model(self, model_name, classified_by_user=False, include_user=False):
        path = sources_base.CLASSIFICATION + "." + model_name

        if classified_by_user:
            return self.db[self.collection].find({path: {"$exists": True}},
                                                 {"_id": 0})

        nested = sources_base.CLASSIFICATION_BY_MODEL + "." + model_name

        if include_user:
            classified_user = list(self.db[self.collection].find({path: {"$exists": True}},
                                                                 {"_id": 0}))
            only_model = list(self.db[self.collection].find({path: {"$exists": False},
                                                             nested: {"$exists": True}},
                                                            {"_id": 0}))
            all_list = classified_user + only_model
            return all_list

        return self.db[self.collection].find({path: {"$exists": False},
                                              nested: {"$exists": True}},
                                             {"_id": 0})


class Model:
    def __init__(self):
        self.db = Database()
        self.collection = DBCollections.MODEL

    # model info will have an identification of the model (given by user)
    # and name of the file of the model
    def save_model(self, model_info):
        return self.db[self.collection].replace_one({model.ID: model_info[model.ID]}, model_info, upsert=True)

    # returns all the saved models
    def get_models(self):
        return self.db[self.collection].find({}, {"_id": 0})

    def get_model(self, model_name):
        return self.db[self.collection].find_one({model.ID: model_name}, {"_id": 0})

    def exist_model(self, model_name):
        return self.db[self.collection].find_one({model.ID: model_name}, {"_id": 0}) is not None


# Creation of a NewsDataBase Instance.
newsDB = RecoveredInfo(DBCollections.NEWS)

# Creation of a TwitterDataBase Instance.
twitterDB = RecoveredInfo(DBCollections.TWITTER)

# Creation of a TwitterDataBase Instance.
redditDB = RecoveredInfo(DBCollections.REDDIT)

# Creation of a ModelDatabase Instance
modelDB = Model()
