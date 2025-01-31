import praw
import os
from dotenv import load_dotenv 


load_dotenv()


reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent="nlp_test-girllanguage"
)


subreddit = reddit.subreddit("AskWomen")
for post in subreddit.hot(limit=5):
    print(f"Title: {post.title}")
    print(f"Text: {post.selftext}\n")
