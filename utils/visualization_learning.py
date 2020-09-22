import math

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib import animation

from classifier.model import DeepLearningModel
from db.database import newsDB
from enums import sources_base
from search import google


def __probability(row, model_name):
    probability = row[model_name][sources_base.CLASSIFICATION_PROBABILITY]
    value = probability if row[model_name][
                               sources_base.CLASSIFICATION_MODEL] == "relevant" else 1 - probability

    return value


def histogram(model_name):
    all_documents = pandas.DataFrame(list(newsDB.get_all_by_model(model_name)))
    all_documents[sources_base.CLASSIFICATION_PROBABILITY] = all_documents[sources_base.CLASSIFICATION_BY_MODEL].apply(
        lambda dict_x: __probability(dict_x, model_name))

    new_df = all_documents.groupby(
        pandas.cut(all_documents[sources_base.CLASSIFICATION_PROBABILITY], np.arange(0, 1.0 + 0.05, 0.05))).count()[
        sources_base.CLASSIFICATION_PROBABILITY]

    new_df.plot.bar()
    plt.show()


def a_histogram(model_name, n_examples=10, n_classified=4):
    def __get_classification(row):
        dict_classification = row[sources_base.CLASSIFICATION]
        return dict_classification[model_name] if model_name in dict_classification else None

    classifier = DeepLearningModel(name=model_name, newModel=False)

    # preparation of the examples documents (already classified by the user)
    documents_classified_by_user = pandas.DataFrame(list(newsDB.get_all_by_model(model_name, classified_by_user=True)))
    documents_classified_by_user[sources_base.CLASSIFICATION] = documents_classified_by_user[
        sources_base.CLASSIFICATION].apply(
        lambda x: x[model_name])

    # preparation of all the documents (including those already classified by the user)
    all_documents = pandas.DataFrame(list(newsDB.get_all_by_model(model_name, include_user=True)))
    all_documents[sources_base.TEXT] = all_documents.apply(lambda x: google.get_text(x), axis=1)
    all_documents[sources_base.CLASSIFICATION] = all_documents.apply(
        lambda x: __get_classification(x), axis=1)
    all_documents = all_documents.drop(
        columns=['url', 'title', 'content_text', 'creation_time', 'comments', 'last_comment', 'classification_by_model',
                 'search_keywords'])

    # this will be use every time that we train the model, its grow with every loop
    # (since we add n_classified docs each time)
    training_docs = pandas.DataFrame(columns=[sources_base.TEXT, sources_base.CLASSIFICATION])

    def __get_examples_doc(number_examples, first=False):
        nonlocal documents_classified_by_user, training_docs

        if first:
            div = number_examples // 2
            examples_d = documents_classified_by_user[
                documents_classified_by_user[sources_base.CLASSIFICATION] == "relevant"].sample(n=math.ceil(div))
            examples_d = examples_d.append(documents_classified_by_user[documents_classified_by_user[
                                                                            sources_base.CLASSIFICATION] == "no_relevant"].sample(
                n=math.floor(div))).apply(
                lambda row: pandas.Series([google.get_text(row), row[google.news.CLASSIFICATION]]), axis=1)
            examples_d.columns = [sources_base.TEXT, sources_base.CLASSIFICATION]
        else:
            n_examples = min(number_examples, len(documents_classified_by_user.index))

            examples_d = documents_classified_by_user.sample(n=n_examples).apply(
                lambda row: pandas.Series([google.get_text(row), row[google.news.CLASSIFICATION]]), axis=1)
            examples_d.columns = [sources_base.TEXT, sources_base.CLASSIFICATION]

        documents_classified_by_user = documents_classified_by_user.drop(examples_d.index)
        training_docs = training_docs.append(examples_d)

    def get_prediction(row):
        nonlocal classifier
        prediction = classifier.predict(row[sources_base.TEXT])
        probability = prediction[sources_base.CLASSIFICATION_PROBABILITY]
        value = probability if prediction[
                                   sources_base.CLASSIFICATION_MODEL] == "__label__relevant" else 1 - probability

        return value

    def __loop(number_examples, n_loop, first=False):
        nonlocal all_documents, training_docs

        __get_examples_doc(number_examples, first)
        classifier.train(training_set=training_docs)
        all_documents["loop" + str(n_loop)] = all_documents.apply(lambda x: get_prediction(x), axis=1)

    # first call
    n_loop = 0
    __loop(n_examples, n_loop=n_loop, first=True)
    # loop-- add n_examples and train classifier
    while len(documents_classified_by_user.index) > 0:
        n_loop += 1
        __loop(n_classified, n_loop=n_loop)

    # in all_documents are all the documents and the probability value of each loop as 'loop#'
    # where # goes from 0 to n_loop

    new_df = pandas.DataFrame()
    for i in range(0, n_loop):
        name_column = "loop" + str(i)
        new_df[name_column] = all_documents.groupby(
            pandas.cut(all_documents[name_column], np.arange(0, 1.0 + 0.05, 0.05))).count()[
            name_column]
    x_label = ["0%-5%", "5%-10%", "10%-15%", "15%-20%", "20%-25%", "25%-30%", "30%-35%", "35%-40%", "40%-45%",
               "45%-50%", "50%-55%", "55%-60%", "60%-65%",
               "65%-70%", "70%-75%", "75%-80%", "80%-85%", "85%-90%", "90%-95%", "95%-100%"]
    new_df["x_label"] = x_label

    fig = plt.figure()
    rects = plt.bar(x=new_df["x_label"], height=new_df["loop0"])

    plt.xlabel('Probabilities Rel-No relevant', fontsize=10)
    plt.ylabel("Number of documents", fontsize=10)
    plt.title('Evolution. Loop 0', fontsize=12)
    plt.xticks(rotation=90)

    def animate(i):
        text = 'Evolution. Loop ' + str(i)
        plt.title(text, fontsize=12)

        name_column = "loop" + str(i)

        for rect, yi in zip(rects, new_df[name_column]):
            rect.set_height(yi)

        return rects

    anim = animation.FuncAnimation(fig, animate, frames=n_loop, interval=1000, blit=True)

    file = model_name + "__evolution.mp4"
    anim.save(file, writer=animation.FFMpegWriter(fps=2))

    #plt.show()


model_name = "KharisseModel"
# histogram(model_name)
a_histogram(model_name, n_examples=6)
