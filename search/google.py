import random
import time

import pandas
from googlesearch import *
from newspaper import Article

from db.database import newsDB
from enums import news

files_folder = "files/"
# user_agent = "Mozilla/5.0"

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'


# TODO correct error message


def search_in_google(keywords):
    print("Searching news with '" + keywords + "' as keywords in Google")
    print("\n")

    list_news = []

    try:
        for result in search_news(keywords, extra_params={'filter': '0'}, user_agent=user_agent):
            print("    >>>  " + result)
            content = get_content(keywords, result)
            if content is not None:
                list_news.append(content)

            time.sleep(random.randint(2, 5))
    except Exception as e:
        print("Error! Too many attempts for today. Please try again tomorrow")
        print("    > " + str(e))

    print("\n")
    print("That's all")

    return list_news


def get_content(keywords, news_url):
    if newsDB.exist_info(news_url):
        news_item = newsDB.get_info(news_url)
        if news.KEYWORDS in news_item:
            if type(news_item[news.KEYWORDS]) == str:
                news_item[news.KEYWORDS] = [news_item[news.KEYWORDS], keywords]
            else:
                news_item[news.KEYWORDS].append(keywords)
        else:
            news_item[news.KEYWORDS] = [keywords]
        newsDB.add_info(news_item)
        return news_item

    try:
        article = Article(news_url)
        article.download()
        article.parse()

        news_item = {
            news.URL: news_url,
            news.TITLE: article.title,
            news.CONTENT: article.text,
            news.CREATION_TIME: article.publish_date,
            news.COMMENTS: [],
            news.LAST_COMMENT: "",
            news.KEYWORDS: [keywords],
            news.sources_base.CLASSIFICATION: {},
            news.sources_base.CLASSIFICATION_BY_MODEL: {}
        }

        newsDB.add_info(news_item)
        print(" ^ " + news_url + " added to db")
        return news_item
    except Exception as e:
        print("Error! Some trouble getting the information of " + news_url)
        print("    > " + str(e))
        return None


def retrieve_news(model_name, list_urls_r, list_urls_i):
    dataframe = pandas.DataFrame(columns=[news.sources_base.TEXT, news.CLASSIFICATION])

    for url in list_urls_r:
        news_item = get_content(model_name, url)
        if news_item is None:
            continue

        news_item[news.CLASSIFICATION][model_name] = "relevant"
        newsDB.add_info(news_item)

        text = get_text(news_item)
        dict = {news.sources_base.TEXT: text, news.CLASSIFICATION: "relevant"}
        dataframe = dataframe.append(dict, ignore_index=True)
        time.sleep(0.5)

    for url in list_urls_i:
        news_item = get_content(model_name, url)
        if news_item is None:
            continue

        news_item[news.CLASSIFICATION][model_name] = "no_relevant"
        newsDB.add_info(news_item)

        text = news_item[news.TITLE] + news_item[news.CONTENT]
        dataframe = dataframe.append({news.sources_base.TEXT: text, news.CLASSIFICATION: "no_relevant"},
                                     ignore_index=True)
        time.sleep(0.5)

    return dataframe


def get_text(document):
    return document[news.TITLE] + " " + document[news.CONTENT]
