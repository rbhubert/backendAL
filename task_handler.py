import pandas

from classifier.model import DeepLearningModel
from enums import model, sources_base
from search import google, reddit, twitter


def train_model(model_info, relevant_docs, irrelevant_docs):
    model_name = model_info["model_name"]
    model_isnew = model_info["model_isnew"]
    model_typedata = model_info["model_type_data"]

    model_t = DeepLearningModel(name=model_name, newModel=model_isnew, dataType=model_typedata)
    type_data = model_t.dataType

    if type_data == model.ModelDataType.GOOGLE:
        training_docs = google.retrieve_news(model_name, relevant_docs, irrelevant_docs)
    elif type_data == model.ModelDataType.REDDIT:
        training_docs = reddit.retrieve_posts(model_name, relevant_docs, irrelevant_docs)
    else:
        training_docs = twitter.retrieve_tweets(model_name, relevant_docs, irrelevant_docs)

    model_t.train(training_set=training_docs)

    return {"message": "Model trained"}


def search_news(model_info, query):
    model_name = model_info["model_name"]
    model_typedata = model_info["model_type_data"]

    model_classifier = DeepLearningModel(name=model_name, newModel=False, dataType=model_typedata)

    documents = google.search_in_google(query)

    for doc in documents:
        if model_name in doc[sources_base.CLASSIFICATION]:  # was already classified by the user
            continue

        # result is a dic { class : , probability : }
        result = model_classifier.predict(google.get_text(doc))
        classification = result[sources_base.CLASSIFICATION_MODEL]
        classification = classification.replace("__label__", "")
        classification = {
            sources_base.CLASSIFICATION_MODEL: classification,
            sources_base.CLASSIFICATION_PROBABILITY: result[sources_base.CLASSIFICATION_PROBABILITY]
        }

        google.newsDB.add_classification_model(doc[google.news.URL], model_name, classification)

    return {"message": "Search done successfully"}


def classify_news(model_name, new_documents):
    for doc in new_documents:
        google.newsDB.add_classification_user(doc[google.news.URL], {model_name: doc[sources_base.CLASSIFICATION]})

    # all documents classified by the user
    all_documents = list(google.newsDB.get_all_by_model(model_name, True))

    training_docs = pandas.DataFrame(columns=[sources_base.TEXT, sources_base.CLASSIFICATION])

    for doc in all_documents:
        text = google.get_text(doc)
        classification = doc[google.news.CLASSIFICATION][model_name]
        dictionary = {sources_base.TEXT: text, sources_base.CLASSIFICATION: classification}
        training_docs = training_docs.append(dictionary, ignore_index=True)

    model_classifier = DeepLearningModel(name=model_name, newModel=False)
    model_classifier.train(training_set=training_docs)

    notc_documents = list(google.newsDB.get_all_by_model(model_name))

    # re classify all documents
    for doc in notc_documents:
        # result is a dic { class : , probability : }
        result = model_classifier.predict(google.get_text(doc))
        classification = result[sources_base.CLASSIFICATION_MODEL]
        classification = classification.replace("__label__", "")
        classification_m = {
            sources_base.CLASSIFICATION_MODEL: classification,
            sources_base.CLASSIFICATION_PROBABILITY: result[sources_base.CLASSIFICATION_PROBABILITY]
        }

        google.newsDB.add_classification_model(doc[google.news.URL], model_name, classification_m)

    return {"message": "Model re-trained"}


def relevant_google(model_name):
    documents = google.newsDB.get_all_by_model(model_name)
    df_documents = __get_relevant_docs(model_name, documents)

    result = []
    for index, document in df_documents.iterrows():
        doc = {
            google.news.URL: document[google.news.URL],
            google.news.TITLE: document[google.news.TITLE],
            google.news.CONTENT: document[google.news.CONTENT]
        }
        result.append(doc)

    return {"documents": result}


def __get_relevant_docs(model_name, documents):
    df_ = pandas.DataFrame(list(documents))
    df_[sources_base.CLASSIFICATION_MODEL] = df_[sources_base.CLASSIFICATION_BY_MODEL].apply(
        lambda dictionary_r: dictionary_r[model_name][sources_base.CLASSIFICATION_MODEL])
    df_[sources_base.CLASSIFICATION_PROBABILITY] = df_[sources_base.CLASSIFICATION_BY_MODEL].apply(
        lambda dictionary_r: dictionary_r[model_name][sources_base.CLASSIFICATION_PROBABILITY])

    df_ = df_[
        (df_[sources_base.CLASSIFICATION_PROBABILITY] > 0.4) & (df_[sources_base.CLASSIFICATION_PROBABILITY] < 0.6)]
    df_ = df_.sort_values([sources_base.CLASSIFICATION_PROBABILITY])

    label = df_.iloc[0][sources_base.CLASSIFICATION_MODEL]
    df1 = df_[df_[sources_base.CLASSIFICATION_MODEL] == label]
    df2 = df_[df_[sources_base.CLASSIFICATION_MODEL] != label]

    desired_sample = 4
    min_sample = min(len(df1), len(df2))
    if df1.empty and df2.empty:
        return pandas.DataFrame()
    elif df1.empty:
        value = min_sample if len(df2) < desired_sample else desired_sample
        return df2.sample(value)
    elif df2.empty:
        value = min_sample if len(df1) < desired_sample else desired_sample
        return df1.sample(value)
    else:
        desired_feach = int(desired_sample / 2)
        value1 = min_sample if len(df1) < desired_feach else desired_feach
        value2 = min_sample if len(df2) < desired_feach else desired_feach

        options1 = df1.sample(value1)
        options2 = df2.sample(value2)
        result = options1.append(options2)
        return result
