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
        classiffication = "relevant" if doc[sources_base.CLASSIFICATION] else "no_relevant"
        google.newsDB.add_classification_user(doc[google.news.URL], {model_name: classiffication})

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

    documents = list(google.newsDB.get_all_by_model(model_name))
    traces = __get_docs_information(model_name, documents)

    return {"message": "Model re-trained", "traces": traces}


def __probability(model_name, dictionary_r):
    probability = dictionary_r[model_name][sources_base.CLASSIFICATION_PROBABILITY]
    value = probability if dictionary_r[model_name][
                               sources_base.CLASSIFICATION_MODEL] == "relevant" else 1 - probability

    return value


def relevant_google(model_name, range):
    range = float(range)
    documents = list(google.newsDB.get_all_by_model(model_name))

    df_ = pandas.DataFrame(documents)
    df_[sources_base.CLASSIFICATION_MODEL] = df_[sources_base.CLASSIFICATION_BY_MODEL].apply(
        lambda dictionary_r: dictionary_r[model_name][sources_base.CLASSIFICATION_MODEL])
    df_[sources_base.CLASSIFICATION_PROBABILITY] = df_[sources_base.CLASSIFICATION_BY_MODEL].apply(
        lambda dictionary_r: __probability(model_name, dictionary_r))

    if range == 0.0:
        df_ = df_[
            (df_[sources_base.CLASSIFICATION_PROBABILITY] < 0.1)]
    elif range == 1.0:
        df_ = df_[
            (df_[sources_base.CLASSIFICATION_PROBABILITY] == 1)]
    else:
        start = range
        end = range + 0.1

        df_ = df_[
            (df_[sources_base.CLASSIFICATION_PROBABILITY] >= start) & (
                    df_[sources_base.CLASSIFICATION_PROBABILITY] < end)]

    df_ = df_.sort_values([sources_base.CLASSIFICATION_PROBABILITY], ascending=False)

    result = []
    for index, document in df_.iterrows():
        url = document[google.news.URL]
        title = document[google.news.TITLE]
        if not title:
            title = url
        content = document[google.news.CONTENT]
        doc = {
            google.news.URL: url,
            google.news.TITLE: title,
            google.news.CONTENT: content,
            sources_base.CLASSIFICATION_BY_MODEL: document[sources_base.CLASSIFICATION_BY_MODEL][model_name],
            "number_words": len(title.split()) + len(content.split())
        }
        result.append(doc)

    return {"documents": result}


def update_info_document(url_doc, title, content):
    pass


def get_info_plot(model_name):
    documents = list(google.newsDB.get_all_by_model(model_name))
    classified, traces = __get_docs_information(model_name, documents)
    return {"classified_by_user": classified, "traces": traces}


def __get_docs_information(model_name, documents):
    df_ = pandas.DataFrame(documents)

    result_trace_model = {
        "x": [],
        "y": [],
        "text": []}

    max_n = 0
    for index, document in df_.iterrows():
        title = document[google.news.TITLE]
        value_probability = document[sources_base.CLASSIFICATION_BY_MODEL][model_name]
        prob = value_probability[sources_base.CLASSIFICATION_PROBABILITY]
        value = prob if value_probability[sources_base.CLASSIFICATION_MODEL] == "relevant" else 1 - prob

        nwords = len(title.split()) + len(document[google.news.CONTENT].split())

        result_trace_model["x"].append(value)
        result_trace_model["y"].append(nwords)
        result_trace_model["text"].append(title)
        max_n = max(max_n, nwords)

    more_documents = list(google.newsDB.get_all_by_model(model_name, classify_by_user=True))
    df_user = pandas.DataFrame(more_documents)

    result_trace_user = {
        "x": [],
        "y": [],
        "text": []
    }

    number_relevants = 0
    number_not_relevants = 0

    relevants = []
    not_relevants = []

    for index, document in df_user.iterrows():
        title = document[google.news.TITLE]
        value_probability = document[sources_base.CLASSIFICATION][model_name]

        doc = {
            "title": title,
            "url": document[google.news.URL]
        }
        if value_probability == "relevant":
            number_relevants = number_relevants + 1
            value = 1
            relevants.append(doc)
        else:
            number_not_relevants = number_not_relevants + 1
            value = 0
            not_relevants.append(doc)

        nwords = len(title.split()) + len(document[google.news.CONTENT].split())

        result_trace_user["x"].append(value)
        result_trace_user["y"].append(nwords)
        result_trace_user["text"].append(title)
        max_n = max(max_n, nwords)

    traces = {
        "trace_model": result_trace_model,
        "trace_user": result_trace_user,
        "max_nwords": max_n,
        "number_model": len(df_.index),
        "number_relevants": number_relevants,
        "number_no_relevants": number_not_relevants
    }

    classified_by_user = {
        "relevants": relevants,
        "not_relevants": not_relevants
    }

    return classified_by_user, traces


def __test_re_classify(model_name):
    classified_by_user = pandas.DataFrame(list(google.newsDB.get_all_by_model(model_name, classify_by_user=True)))
    training_docs = classified_by_user.apply(
        lambda row: pandas.Series([google.get_text(row), row[google.news.CLASSIFICATION][model_name]]), axis=1)

    classifier = DeepLearningModel(name=model_name, newModel=False)
    # if you need to train your model with different parameters, you can change them here
    # epoch = 50, lr = 1.0, wordNgram = 1
    classifier.train(training_set=training_docs)

    docs_no_classified = list(google.newsDB.get_all_by_model(model_name))

    # re classify all documents
    for doc in docs_no_classified:
        result = classifier.predict(google.get_text(doc))
        classification = result[sources_base.CLASSIFICATION_MODEL]
        classification = classification.replace("__label__", "")
        classification_m = {
            sources_base.CLASSIFICATION_MODEL: classification,
            sources_base.CLASSIFICATION_PROBABILITY: result[sources_base.CLASSIFICATION_PROBABILITY]
        }

        google.newsDB.add_classification_model(doc[google.news.URL], model_name, classification_m)


# __test_re_classify("KharisseModel")


def __test_accuracy(model_name):
    classifier = DeepLearningModel(name=model_name, newModel=False)
    classified_by_user = google.newsDB.get_all_by_model(model_name, classify_by_user=True)

    total_docs = classified_by_user.count()
    classified_correctly = 0

    for document in classified_by_user:
        document_text = google.get_text(document)
        classification_user = document["classification_user"][model_name]
        classification_model = classifier.predict(document_text)
        classification_model_text = classification_model["classification_value"].replace("__label__", "")

        if classification_user == classification_model_text:
            classified_correctly += 1

    print("total docs:", total_docs)
    print("classified correctly: ", classified_correctly)
    print("accuracy: ", total_docs / classified_correctly)

# __test_accuracy("KharisseModel")
