# coding=utf-8
import json
import os

import pandas
from flask import Flask, Response
from flask import request
from flask_cors import CORS

from classifier.model import DeepLearningModel
from db.database import modelDB
from enums import model, sources_base
from enums.config import DATABASE
from search import google, reddit, twitter
FLASK_APP = Flask(__name__)
CORS(FLASK_APP)  # allowing request from different urls... (localhost in another port)

# just to avoid a windows error... o.ó
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')





@FLASK_APP.route('/models')
def get_models():
    models = modelDB.get_models()

    models_names = []
    for m in models:
        models_names.append(m[model.ORIGINAL_NAME])

    respjs = json.dumps({"models": models_names})
    return Response(respjs, status=200, mimetype='application/json')


@FLASK_APP.route('/model/<model_name>')
def get_model_info(model_name):
    model_id = model_name.lower().replace(" ", "_")
    model_info = modelDB.get_model(model_id)
    respjs = json.dumps({"model_info": model_info})
    return Response(respjs, status=200, mimetype='application/json')


@FLASK_APP.route('/train', methods=['POST'])
def train_model():
    if request.method == 'POST':
        model_info = json.loads(request.form.get('model_info'))
        relevant_docs = request.form.get('relevant_docs').split()
        irrelevant_docs = request.form.get('irrelevant_docs').split()

        model_name = model_info["model_name"]
        model_isnew = model_info["model_isnew"]
        model_typedata = model_info["model_type_data"]

        model_t = DeepLearningModel(name=model_name, newModel=model_isnew, dataType=model_typedata)
        type_data = model_t.dataType

        if type_data == model.ModelDataType.GOOGLE:
            training_docs = google.retrieve_news(relevant_docs, irrelevant_docs)
        elif type_data == model.ModelDataType.REDDIT:
            training_docs = reddit.retrieve_posts(relevant_docs, irrelevant_docs)
        else:
            training_docs = twitter.retrieve_tweets(relevant_docs, irrelevant_docs)

        model_t.train(training_set=training_docs)

        # respjs = json.dumps(models_names)
        return Response("everythinfineee wey", status=200, mimetype='application/json')

    return Response("pos nos e que paso", status=200, mimetype='application/json')


@FLASK_APP.route('/search/google', methods=['POST'])
def search_news():
    query = request.form.get('query')

    model_info = json.loads(request.form.get('model_info'))
    model_name = model_info["model_name"]
    model_typedata = model_info["model_type_data"]

    model_classifier = DeepLearningModel(name=model_name, newModel=False, dataType=model_typedata)

    # TODO el verdadero debería ser search in google
    # documents = google.search_in_google(query)

    documents = google.newsDB.get_all_truchis()

    for doc in documents:
        result = model_classifier.predict(google.get_text(doc))

        # result is a dic { class : , probability : }
        # doc[sources_base.CLASSIFICATION_BY_MODEL][model_name] = {
        #     sources_base.CLASSIFICATION_MODEL: result[sources_base.CLASSIFICATION_MODEL].replace("__label__", ""),
        #     sources_base.CLASSIFICATION_PROBABILITY: result[sources_base.CLASSIFICATION_PROBABILITY]
        # }

        classification = {
            sources_base.CLASSIFICATION_MODEL_NAME: model_name,
            sources_base.CLASSIFICATION_MODEL: result[sources_base.CLASSIFICATION_MODEL].replace("__label__", ""),
            sources_base.CLASSIFICATION_PROBABILITY: result[sources_base.CLASSIFICATION_PROBABILITY]
        }

        google.newsDB.add_classification_model(doc[google.news.URL], model_name, classification)
        # google.newsDB.add_info(doc)

    return Response("everythinfineee wey", status=200, mimetype='application/json')


@FLASK_APP.route('/relevant/google/<model_name>')
def get_relevant_news(model_name):
    documents = google.newsDB.get_all_by_model(model_name)
    df_documents = __get_relevants_docs(documents)

    result = []
    for index, document in df_documents.iterrows():
        doc = {
            google.news.URL: document[google.news.URL],
            google.news.TITLE: document[google.news.TITLE],
            google.news.CONTENT: document[google.news.CONTENT]
        }
        result.append(doc)

    respjs = json.dumps({"documents": result})
    return Response(respjs, status=200, mimetype='application/json')


@FLASK_APP.route('/classify/google/<model_name>', methods=['POST'])
def classify_news(model_name):
    new_documents = json.loads(request.form.get('documents'))
    for doc in new_documents:
        google.newsDB.add_classification_user(doc[google.news.URL], doc[sources_base.CLASSIFICATION])

    # all documents classified by the user
    all_documents = list(google.newsDB.get_all_by_model(model_name, True))

    training_docs = pandas.DataFrame(columns=[sources_base.TEXT, sources_base.CLASSIFICATION])

    for doc in all_documents:
        text = google.get_text(doc)
        classification = doc[google.news.CLASSIFICATION]
        dictionary = {sources_base.TEXT: text, sources_base.CLASSIFICATION: classification}
        training_docs = training_docs.append(dictionary, ignore_index=True)

    model_classifier = DeepLearningModel(name=model_name, newModel=False)
    model_classifier.train(training_set=training_docs)

    notc_documents = list(google.newsDB.get_all_by_model(model_name))

    for doc in notc_documents:
        result = model_classifier.predict(google.get_text(doc))
        # result is a dic { class : , probability : }
        # doc[sources_base.CLASSIFICATION_BY_MODEL][model_name] = {
        #     sources_base.CLASSIFICATION_MODEL: result[sources_base.CLASSIFICATION_MODEL].replace("__label__", ""),
        #     sources_base.CLASSIFICATION_PROBABILITY: result[sources_base.CLASSIFICATION_PROBABILITY]
        # }

        classification_m = {
            sources_base.CLASSIFICATION_MODEL_NAME: model_name,
            sources_base.CLASSIFICATION_MODEL: result[sources_base.CLASSIFICATION_MODEL].replace("__label__", ""),
            sources_base.CLASSIFICATION_PROBABILITY: result[sources_base.CLASSIFICATION_PROBABILITY]
        }
        # google.newsDB.add_info(doc)
        google.newsDB.add_classification_model(doc[google.news.URL], model_name, classification_m)

    return Response("everythinfineee wey", status=200, mimetype='application/json')


def __get_relevants_docs(documents):
    df_ = pandas.DataFrame(list(documents))
    df_[sources_base.CLASSIFICATION_MODEL] = df_[sources_base.CLASSIFICATION_BY_MODEL].apply(
        lambda dictionary_r: dictionary_r[sources_base.CLASSIFICATION_MODEL])
    df_[sources_base.CLASSIFICATION_PROBABILITY] = df_[sources_base.CLASSIFICATION_BY_MODEL].apply(
        lambda dictionary_r: dictionary_r[sources_base.CLASSIFICATION_PROBABILITY])

    df_ = df_[
        (df_[sources_base.CLASSIFICATION_PROBABILITY] > 0.4) & (df_[sources_base.CLASSIFICATION_PROBABILITY] < 0.6)]
    df_ = df_.sort_values([sources_base.CLASSIFICATION_PROBABILITY])

    label = df_.iloc[0][sources_base.CLASSIFICATION_MODEL]
    df1 = df_[df_[sources_base.CLASSIFICATION_MODEL] == label]
    options1 = df1.sample(2)
    df2 = df_[df_[sources_base.CLASSIFICATION_MODEL] != label]
    options2 = df2.sample(2)

    result = options1.append(options2)
    return result


if __name__ == '__main__':
    FLASK_APP.run()

# @FLASK_APP.route('/search/google/<keywords>')
# def search_news(keywords):
#     search_in_google(keywords)
#     # response = tasks_requestHandler.remove_account(socialMedia, username)
#     # respjs = json.dumps(response)
#     # return Response(respjs, status=200, mimetype='application/json')
#     return Response({}, status=200, mimetype='application/json')
#
#
# @FLASK_APP.route('/search/reddit/<keywords>')
# def search_posts(keywords):
#     search_in_reddit(keywords)
#     return Response({}, status=200, mimetype='application/json')
#
#
# @FLASK_APP.route('/search/twitter/<keywords>')
# def search_tweets(keywords):
#     search_in_twitter(keywords)
#     return Response({}, status=200, mimetype='application/json')
#
#
# @FLASK_APP.route('/train/google/')
# def train_news():
#     pass
#
#
# @FLASK_APP.route('/train/reddit/')
# def train_posts():
#     pass
#
#
# @FLASK_APP.route('/train/twitter/')
# def train_tweets():
#     pass
#
#
# @FLASK_APP.route('/relevant/google/')
# def get_relevant_news():
#     pass
#
#
# @FLASK_APP.route('/relevant/reddit/')
# def get_relevant_posts():
#     pass
#
#
# @FLASK_APP.route('/relevant/twitter/')
# def get_relevant_tweets():
#     pass
#
#
# @FLASK_APP.route('/classify/google/<news>')
# def classify_news(news):
#     pass
#
#
# @FLASK_APP.route('/classify/reddit/<posts>')
# def classify_posts(posts):
#     pass
#
#
# @FLASK_APP.route('/classify/twitter/<tweets>')
# def classify_tweets(tweets):
#     pass
