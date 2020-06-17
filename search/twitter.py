import tweepy
from twarc import Twarc

from db.database import twitterDB
from enums import tweets
from enums.config import *

auth = tweepy.OAuthHandler(TWITTER_AUTH[0], TWITTER_AUTH[1])
auth.set_access_token(TWITTER_ACCESS_TOKEN[0], TWITTER_ACCESS_TOKEN[1])

# api rest
twitterAPI = tweepy.API(auth, wait_on_rate_limit=True,
                        wait_on_rate_limit_notify=True)
# api for comments
twitterReplies = Twarc(TWITTER_AUTH[0], TWITTER_AUTH[1],
                       TWITTER_ACCESS_TOKEN[0], TWITTER_ACCESS_TOKEN[1])


def search_in_twitter(keywords):
    print("Searching tweets with '" + keywords + "' as keywords in Twitter")
    print("\n")

    all_tweets = twitterAPI.search(q=keywords)
    for tweet in all_tweets:
        print("    >>>  " + tweet.text)
        get_content(tweet)

    print("\n")
    print("That's all")


def get_content(tweet):
    try:
        tweet_id = tweet.id_str
        if twitterDB.exist_tweet(tweet_id):
            return

        tweet_item = get_dic_of_tweet(tweet._json, False)
        twitterDB.add_tweet(tweet_item)

    except Exception as e:
        print("Error! Some trouble getting the information of the tweet")
        print("    > " + str(e))


def get_replies(tweet):
    all_replies = twitterReplies.replies(tweet, False)
    replies = []

    for reply_x in all_replies:
        if reply_x["id_str"] != tweet["id_str"]:
            reply = get_dic_of_tweet(reply_x, True)
            replies.append(reply)

    return replies


def get_dic_of_tweet(tweet, is_comment):
    dic_tweet = {
        tweets.TWEET_ID: tweet["id_str"],
        tweets.USER: {
            tweets.User.USER_ID: tweet["user"]["id_str"],
            tweets.User.USERNAME: tweet["user"]["screen_name"]
        },
        tweets.CONTENT: tweet["text"],
        tweets.HASHTAGS: tweet["entities"]["hashtags"],
        tweets.USER_MENTIONS: [{
            "user_id": user["id_str"],
            "username": user["screen_name"]
        } for user in tweet["entities"]["user_mentions"]],
        tweets.CREATION_TIME: tweet["created_at"],
        tweets.IS_RETWEETED: True if "retweeted_status" in tweet else False,
        tweets.FAVORITES: tweet["favorite_count"],
        tweets.RETWEETS: tweet["retweet_count"],
        tweets.COMMENTS: [] if is_comment else get_replies(tweet)
    }

    return dic_tweet


def retrieve_tweets(model_name, list_tweets_r, list_tweets_i):
    pass
