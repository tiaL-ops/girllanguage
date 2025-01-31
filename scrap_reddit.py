import praw
import os
import pandas as pd
import time
from dotenv import load_dotenv


load_dotenv()


reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent="nlp_test-girllanguage"
)


try:
    search_results = reddit.subreddit("relationships").search(
        query="does he love me?", limit=10, sort="new"
    )

    data = []
    for post in search_results:
      
        text = (post.title or "") + " " + (post.selftext or "")
        if text.strip():  
            data.append({"text": text, "label": 1}) 

        post.comments.replace_more(limit=0)  
        comments = [comment.body for comment in post.comments[:5]]  
        
        for comment in comments:
            if comment.strip(): 
                data.append({"post_text": text, "comment": comment, "label": 1}) 
        time.sleep(1)  

    
    if data:
        df = pd.DataFrame(data)
        df.to_csv("reddit_data.csv", index=False)
        print("✅ Reddit data saved: reddit_data.csv")
    else:
        print("⚠️ No valid data found.")

except Exception as e:
    print(f"❌ Error: {e}")
