import praw

from db.database import redditDB
from enums import reddit

reddit_api = praw.Reddit(user_agent="Posts Extraction (by /u/iaimwithmyeye)",
                         client_id="mxkLdSHD57yaow", client_secret="ZRvBOpqSGCjeGzWSlEByXCe4oOs")


def search_in_reddit(keywords):
    print("Searching reddit posts with '", keywords, "' as keywords")
    print("\n")

    all_subreddit = reddit_api.subreddit("all")
    for post in all_subreddit.search(query=keywords):
        print("    >>>  " + post.title)
        get_content(post)

    print("\n")
    print("That's all")


def get_content(post):
    permanent_link = post.permalink
    if redditDB.exist_post(permanent_link):
        return

    reddit_post = get_dic_post(post, False)
    redditDB.add_post(reddit_post)


def get_comments(post, is_reply):
    comments = []

    if is_reply:
        comment_forest = post.replies
    else:
        comment_forest = post.comments

    comment_forest.replace_more()

    for comment_x in comment_forest:
        comment = get_dic_post(comment_x, True)
        comments.append(comment)

    return comments


def get_dic_post(post, is_reply):
    reddit_post = {
        reddit.CREATION_TIME: post.created_utc,
        reddit.USER: '[deleted]' if not post.author else post.author.name,
        reddit.CONTENT: post.body if is_reply else post.selftext,
        reddit.UPVOTES: post.score,
        reddit.COMMENTS: get_comments(post, is_reply)
    }

    if not is_reply:
        reddit_post[reddit.TITLE] = post.title
        reddit_post[reddit.PERMANENT_LINK] = post.permalink

    return reddit_post


def retrieve_posts(list_posts_r, list_posts_i):
    pass