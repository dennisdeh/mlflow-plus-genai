import json
import praw
from typing import List, Dict, Any


class RedditScraper:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        username: str = None,
        password: str = None,
    ):
        """
        Initializes the Reddit API connection.

        Args:
            client_id (str): Reddit app client ID.
            client_secret (str): Reddit app client secret.
            user_agent (str): A descriptive name for your application.
            username (str, optional): Reddit account username.
            password (str, optional): Reddit account password.
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username,
            password=password,
        )

    def scrape_subreddit_comments(
        self,
        subreddit_name: str = "SwissPersonalFinance",
        post_limit: int = 5,
        comment_limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Scrapes posts and their top comments from a specific subreddit.

        Args:
            subreddit_name (str): Name of the subreddit to scrape.
            post_limit (int): Number of posts to retrieve from the 'hot' section.
            comment_limit (int): Number of top-level comments to retrieve per post.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing post data and top comments.
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        scraped_data = []

        # Fetching hot posts
        for submission in subreddit.hot(limit=post_limit):
            # Ensure we are getting the best comments
            submission.comment_sort = "top"
            submission.comments.replace_more(
                limit=0
            )  # Flatten comment tree and remove "load more"

            comments = []
            for comment in submission.comments[:comment_limit]:
                comments.append(
                    {
                        "comment_id": comment.id,
                        "body": comment.body,
                        "score": comment.score,
                    }
                )

            scraped_data.append(
                {
                    "post_id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "score": submission.score,
                    "top_comments": comments,
                }
            )

        return scraped_data

    def format_for_llm(self, data: List[Dict[str, Any]]) -> str:
        """
        Formats the scraped data into a JSON string structured for LLM fine-tuning or RAG.
        Typically uses a 'prompt' and 'completion' or 'instruction'/'input'/'output' pattern.
        """
        llm_formatted_data = []

        for entry in data:
            # Combining post content for context
            context = f"Title: {entry['title']}\nContent: {entry['selftext']}"

            # Creating a training/inference example for each top comment
            for i, comment in enumerate(entry["top_comments"]):
                llm_formatted_data.append(
                    {
                        "instruction": f"Based on the following Reddit post from r/SwissPersonalFinance, provide a helpful community response.",
                        "input": context,
                        "output": comment["body"],
                        "metadata": {
                            "post_id": entry["post_id"],
                            "comment_rank": i + 1,
                            "score": comment["score"],
                        },
                    }
                )

        return json.dumps(llm_formatted_data, indent=4)
