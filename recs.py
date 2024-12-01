#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install implicit scikit-learn surprise


# In[2]:


#get_ipython().system('pip uninstall numpy -y')
#get_ipython().system("pip install 'numpy<2.0'")
#get_ipython().system('pip install --force-reinstall surprise scikit-surprise')
#

# In[3]:


import pandas as pd
import numpy as np
import random
import pickle

from itertools import chain
from implicit.bpr import BayesianPersonalizedRanking

from numpy import asarray
from numpy import savetxt

from scipy import sparse
from sklearn.model_selection import train_test_split

from surprise import Dataset
from surprise import Reader
from surprise import SVD, SVDpp, NMF

from surprise.model_selection import cross_validate
from surprise.dump import dump

from re import U
from bs4 import BeautifulSoup
from pymongo.operations import ReplaceOne
import requests

import nest_asyncio
import asyncio
from aiohttp import ClientSession

nest_asyncio.apply()

import pymongo
from pymongo import UpdateOne, ReplaceOne
from pymongo.errors import BulkWriteError

import datetime

from pprint import pprint

import os


# In[ ]:


def mean_reciprocal_rank(recs, test):
    """
    Compute the Mean Reciprocal Rank (MRR).

    Parameters:
        recs (list): List of tuples (movie_id, score) in ranked order.
        test (list): List of tuples (movie_id, rating).

    Returns:
        float: Mean Reciprocal Rank (MRR).
    """
    test_ids = {movie_id for movie_id, _ in test}
    for rank, (movie_id, _) in enumerate(recs, start=1):
        if movie_id in test_ids:
            return 1 / rank
    return 0  # No relevant items found in the recommendations

def ndcg(recs, test):
    """
    Compute the Normalized Discounted Cumulative Gain (nDCG).

    Parameters:
        recs (list): List of tuples (movie_id, score) in ranked order.
        test (list): List of tuples (movie_id, rating).

    Returns:
        float: Normalized Discounted Cumulative Gain (nDCG).
    """
    test_ratings = dict(test)

    # Compute DCG
    dcg = sum(test_ratings.get(movie_id, 0) / np.log2(rank + 1) for rank, (movie_id, _) in enumerate(recs, start=1))

    # Compute Ideal DCG
    ideal_ranking = sorted(test_ratings.items(), key=lambda x: x[1], reverse=True)
    idcg = sum(rating / np.log2(rank + 1) for rank, (movie_id, rating) in enumerate(ideal_ranking, start=1))

    return dcg / idcg if idcg > 0 else 0

def precision_at_k(recs, test, k=50):
    """
    Compute Precision at k (P@k).

    Parameters:
        recs (list): List of tuples (movie_id, score) in ranked order.
        test (list): List of tuples (movie_id, rating).
        k (int): The number of top recommendations to consider.

    Returns:
        float: Precision at k.
    """
    test_ids = {movie_id for movie_id, _ in test}
    top_k_recs = recs[:k]
    hits = len([movie_id for movie_id, _ in top_k_recs if movie_id in test_ids])
    return hits / k if k > 0 else 0

def compute_metrics(recs, test, k=10):
    """
    Compute evaluation metrics.

    Parameters:
        recs (list): List of tuples (movie_id, score) in ranked order.
        test (list): List of tuples (movie_id, rating).
        k (int): The number of top recommendations to consider for P@k.

    Returns:
        dict: Dictionary of computed metrics.
    """
    metrics = {}
    metrics['precision_at_k'] = precision_at_k(recs, test, k=k)
    metrics['mean_reciprocal_rank'] = mean_reciprocal_rank(recs, test)
    metrics['ndcg'] = ndcg(recs, test)
    return metrics


# In[ ]:


movies = pd.read_csv('movie_data.csv', quotechar='"', escapechar="\\", on_bad_lines='skip', engine="python")
users = pd.read_csv('users_export.csv')
ratings = pd.read_csv('training_data.csv', nrows=1000000)
ratings = ratings[['user_id', 'movie_id', 'rating_val']]


# In[ ]:


def create_movie_data_sample(movies, movie_list):
    movie_df = movies[movies["movie_id"].isin(movie_list)].copy()
    movie_df = movie_df[["movie_id", "image_url", "movie_title", "year_released"]]
    movie_df["image_url"] = (
        movie_df["image_url"]
        .fillna("")
        .str.replace("https://a.ltrbxd.com/resized/", "", regex=False)
    )
    movie_df["image_url"] = (
        movie_df["image_url"]
        .fillna("")
        .str.replace(
            "https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png",
            "",
            regex=False,
        )
    )

    return movie_df

# Generate training data sample
training_df = ratings.copy()

review_counts_df = (
    ratings.groupby("movie_id")
    .size()  # Count reviews per movie
    .reset_index(name="count")  # Rename the count column
)

threshold_movie_list = review_counts_df["movie_id"].to_list()

# Generate movie data CSV
movie_df = create_movie_data_sample(movies, threshold_movie_list)

# Use movie_df to remove any items from threshold_list that do not have a "year_released"
# This virtually always means it's a collection of more popular movies (such as the LOTR trilogy) and we don't want it included in recs
retain_list = movie_df.loc[
    (movie_df["year_released"].notna() & movie_df["year_released"] != 0.0)
]["movie_id"].to_list()

threshold_movie_list = [x for x in threshold_movie_list if x in retain_list]


# In[ ]:


async def fetch(url, session, input_data={}):
    async with session.get(url) as response:
        try:
            return await response.read(), input_data
        except:
            return None, None

async def generate_ratings_operations(response, send_to_db=True, return_unrated=False):
    # Parse ratings page response for each rating/review, use lxml parser for speed
    soup = BeautifulSoup(response[0], "lxml")
    reviews = soup.findAll("li", attrs={"class": "poster-container"})

    # Create empty array to store list of bulk operations or rating objects
    ratings_operations = []
    movie_operations = []

    for review in reviews:
        movie_id = review.find("div", attrs={"class", "film-poster"})[
            "data-target-link"
        ].split("/")[-2]

        # Check for rating
        rating = review.find("span", attrs={"class": "rating"})
        if not rating:
            if not return_unrated:
                continue
            rating_val = -1
        else:
            rating_class = rating["class"][-1]
            rating_val = int(rating_class.split("-")[-1])

        rating_object = {
            "movie_id": movie_id,
            "rating_val": rating_val,
            "user_id": response[1]["username"],
        }

        if not send_to_db:
            ratings_operations.append(rating_object)
        else:
            # Add UpdateOne operations for database insertion
            ratings_operations.append(
                UpdateOne(
                    {"user_id": response[1]["username"], "movie_id": movie_id},
                    {"$set": rating_object},
                    upsert=True,
                )
            )
            movie_operations.append(
                UpdateOne(
                    {"movie_id": movie_id},
                    {"$set": {"movie_id": movie_id}},
                    upsert=True,
                )
            )

    return ratings_operations, movie_operations

def build_model(df, user_data, model='SVD', num_factors=10, learning_rate=0.01, regularization=0.05, iterations=100):
    import random
    import numpy as np
    from scipy.sparse import csr_matrix
    from implicit.bpr import BayesianPersonalizedRanking

    # Set random seed
    random.seed(12)
    np.random.seed(12)

    # Filter user_data based on the model type
    if model == 'BPR':
        # Include both rated items and likes for BPR
        user_data_filtered = user_data
    else:
        # Exclude likes for SVD and NMF
        user_data_filtered = [x for x in user_data if x['rating_val'] > 0]

    # Convert filtered user_data to a DataFrame and append it to the existing data
    user_df = pd.DataFrame(user_data_filtered)
    df = pd.concat([df, user_df]).reset_index(drop=True)
    df.drop_duplicates(inplace=True)

    if model == 'BPR':
        # Add likes to dataset
        likes = pd.read_csv('likes.csv', quotechar='"', escapechar="\\", on_bad_lines='skip', engine="python")
        likes['rating_val'] = -1
        df = pd.concat([df, likes[['movie_id', 'user_id', 'rating_val']]], ignore_index=True)
        df = df.drop_duplicates(subset=['movie_id', 'user_id'], ignore_index=True)

        # Filter movies with few interactions
        movie_threshold = 5
        df = df[df['movie_id'].isin(df.groupby('movie_id').size()[lambda x: x >= movie_threshold].index)]

        user_mapping = {id: index for index, id in enumerate(df['user_id'].unique())}
        movie_mapping = {id: index for index, id in enumerate(df['movie_id'].unique())}
        reverse_movie_mapping = {index: id for id, index in movie_mapping.items()}

        df['user_idx'] = df['user_id'].map(user_mapping)
        df['movie_idx'] = df['movie_id'].map(movie_mapping)
        df['rating_val'] = 1  # Set all interactions to 1 for implicit feedback

        sparse_matrix = csr_matrix(
            (df['rating_val'], (df['user_idx'], df['movie_idx'])),
            shape=(df['user_idx'].max() + 1, df['movie_idx'].max() + 1)
        )

        algo = BayesianPersonalizedRanking(factors=num_factors, learning_rate=learning_rate,
                                           regularization=regularization, random_state=42)
        algo.fit(sparse_matrix.T)

        bpr_data = (user_mapping, movie_mapping, reverse_movie_mapping, sparse_matrix)

    else:
        # Surprise model fallback for SVD or NMF
        from surprise import SVD, NMF, Dataset, Reader
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(df[["user_id", "movie_id", "rating_val"]], reader)
        algo = NMF(random_state=42) if model == 'NMF' else SVD(random_state=42)
        trainingSet = data.build_full_trainset()
        algo.fit(trainingSet)
        bpr_data = None

    user_watched_list = [x['movie_id'] for x in user_data_filtered]

    return algo, user_watched_list, bpr_data

def get_page_count(username):
    url = "https://letterboxd.com/{}/films/by/date"
    r = requests.get(url.format(username))

    soup = BeautifulSoup(r.text, "lxml")

    body = soup.find("body")

    try:
        if "error" in body["class"]:
            return -1, None
    except KeyError:
        print(body)
        return -1, None

    try:
        page_link = soup.findAll("li", attrs={"class", "paginate-page"})[-1]
        num_pages = int(page_link.find("a").text.replace(",", ""))
        display_name = (
            body.find("section", attrs={"class": "profile-header"})
            .find("h1", attrs={"class": "title-3"})
            .text.strip()
        )
    except IndexError:
        num_pages = 1
        display_name = None

    return num_pages, display_name

async def get_user_ratings(
    username,
    db_cursor=None,
    mongo_db=None,
    store_in_db=True,
    num_pages=None,
    return_unrated=False,
):
    url = "https://letterboxd.com/{}/films/by/date/page/{}/"

    if not num_pages:
        user = db_cursor.find_one({"username": username})
        num_pages = user["recent_page_count"]

    async with ClientSession() as session:
        tasks = [
            asyncio.ensure_future(
                fetch(url.format(username, i + 1), session, {"username": username})
            )
            for i in range(num_pages)
        ]
        scrape_responses = await asyncio.gather(*tasks)
        scrape_responses = [x for x in scrape_responses if x]

    tasks = [
        asyncio.ensure_future(
            generate_ratings_operations(
                response, send_to_db=store_in_db, return_unrated=return_unrated
            )
        )
        for response in scrape_responses
    ]
    parse_responses = await asyncio.gather(*tasks)

    if not store_in_db:
        # Flatten the raw data into a single list
        parse_responses = list(
            chain.from_iterable(list(chain.from_iterable(parse_responses)))
        )
        return parse_responses

    upsert_ratings_operations = []
    upsert_movies_operations = []
    for response in parse_responses:
        upsert_ratings_operations += response[0]
        upsert_movies_operations += response[1]

    return upsert_ratings_operations, upsert_movies_operations

async def get_user_data(username, data_opt_in=False):
    num_pages, display_name = get_page_count(username)

    if num_pages == -1:
        return [], "user_not_found"

    user_ratings = await get_user_ratings(
        username,
        db_cursor=None,
        mongo_db=None,
        store_in_db=False,  # Ensure we get raw data
        num_pages=num_pages,
        return_unrated=True,
    )

    # Filter out items where no rating or like is present
    user_ratings = [x for x in user_ratings if x["rating_val"] >= 0 or x["rating_val"] == -1]

    if data_opt_in:
        send_to_db(username, display_name, user_ratings=user_ratings)

    return user_ratings, "success"

df = training_df.copy()

"""
user_data, status = await get_user_data('geraldne')

if status == "success":
    user_data_train, user_data_test = train_test_split(
                user_data, test_size=0.2, random_state=42, stratify=[val == -1 for val in rating_vals])
    algo, user_watched_list, bpr_data = build_model(df, user_data_train, model='BPR')
"""


# In[ ]:


from collections import defaultdict

try:
    from .db_config import config
except ImportError:
    config = None


def get_top_n(predictions, n=20):
    top_n = [(iid, est) for uid, iid, true_r, est, _ in predictions]
    top_n.sort(key=lambda x: (x[1], random.random()), reverse=True)

    return top_n[:n]


def run_model(username, algo, user_watched_list, user_data_test, threshold_movie_list, bpr_data=None, num_recommendations=100):

    user_watched_test = [(x['movie_id'], x['rating_val']) for x in user_data_test]

    unwatched_movies = [x for x in threshold_movie_list if x not in user_watched_list]

    if bpr_data:
        user_mapping, movie_mapping, reverse_movie_mapping, sparse_matrix = bpr_data
        if username in user_mapping:
            user_idx = user_mapping[username]
            if 0 <= user_idx < sparse_matrix.shape[0]:
                user_items = sparse_matrix[user_idx]
                recommendations = algo.recommend(user_idx, user_items, N=num_recommendations, filter_already_liked_items=False)
                item_indices, scores = recommendations
                top_n = [(reverse_movie_mapping[item_idx], score) for item_idx, score in zip(item_indices, scores) if item_idx in reverse_movie_mapping]
                top_n = sorted(top_n, key=lambda x: x[1], reverse=True)
            else:
                raise IndexError(f"user_idx {user_idx} out of bounds for sparse_matrix with shape {sparse_matrix.shape}.")
        else:
            raise ValueError(f"username '{username}' not found in user_mapping.")
    else:
        predictions = algo.test([(username, x, 0) for x in unwatched_movies])
        top_n = get_top_n(predictions, num_recommendations)

    movie_fields = ["image_url", "movie_id", "movie_title", "year_released", "genres", "original_language", "popularity", "runtime", "release_date"]
    metrics = compute_metrics(top_n, user_watched_test)
    movie_ids = [x[0] for x in top_n]
    filtered_movies = movies[movies["movie_id"].isin(movie_ids)]
    movie_data = {
        row["movie_id"]: {k: row[k] for k in filtered_movies.columns if k in movie_fields}
        for _, row in filtered_movies.iterrows()
    }

    return_object = [
        {"movie_id": x[0], "predicted_rating": round(x[1], 3), "movie_data": movie_data[x[0]]}
        for x in top_n if x[0] in movie_data.keys()
    ]
    return return_object, metrics

#recs, metrics = run_model('geraldne', algo, user_watched_list, user_data_test, threshold_movie_list, bpr_data, 10)


# In[ ]:


async def fetch_all_user_data(users_to_test, get_user_data):
    """
    Fetch data for all users asynchronously and store it in a dictionary.

    Parameters:
        users_to_test (list): List of usernames to fetch data for.
        get_user_data (function): Asynchronous function to fetch user data.

    Returns:
        user_data_dict (dict): Dictionary with username as key and user data as value.
    """
    user_data_dict = {}

    async def fetch_user(username):
        """Fetch data for a single user."""
        print(f"Fetching data for user: {username}")
        user_data, status = await get_user_data(username)
        if status and user_data:
            user_data_dict[username] = user_data
        else:
            print(f"Failed to fetch data for user: {username}. Skipping.")

    # Create a list of tasks to fetch data for all users concurrently
    tasks = [fetch_user(username) for username in users_to_test]
    await asyncio.gather(*tasks)

    return user_data_dict

def save_user_data(user_data_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(user_data_dict, f)
    print(f"User data saved to {filename}")

def load_user_data(filename):
    with open(filename, 'rb') as f:
        user_data_dict = pickle.load(f)
    print(f"User data loaded from {filename}")
    return user_data_dict

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def evaluate_models(users_to_test, user_data_dict, df, threshold_movie_list):
    results = []
    recommendations = {}

    def process_user(username):
        user_results = []
        user_recommendations = {}

        if username not in user_data_dict:
            print(f"No data available for user: {username}. Skipping.")
            return None

        try:
            # Extract user-specific data and split into train/test
            user_data = user_data_dict[username]
            rating_vals = [x['rating_val'] for x in user_data]
            user_data_train, user_data_test = train_test_split(
                user_data, test_size=0.2, random_state=42, stratify=[val == -1 for val in rating_vals]
            )
            user_watched_test = [(x['movie_id'], x['rating_val']) for x in user_data_test]

            # Initialize the user's entry in recommendations
            user_recommendations = {}

        except Exception as e:
            print(f"Error splitting data for user '{username}': {e}")
            return None

        # Train and evaluate each model
        for model_name in ['SVD', 'NMF', 'BPR']:
            try:
                # Build the model and retrieve necessary data
                algo, user_watched_list, bpr_data = build_model(df, user_data_train, model=model_name)

                # Run the model and retrieve recommendations and metrics
                recs, metrics = run_model(
                    username,
                    algo,
                    user_watched_list,
                    user_data_test,
                    threshold_movie_list,
                    bpr_data,
                    num_recommendations=50
                )

                # Add user and model information to metrics
                metrics['model'] = model_name
                metrics['user'] = username
                user_results.append(metrics)

                # Store recommendations
                user_recommendations[model_name] = recs

            except ValueError as ve:
                print(f"Skipping user '{username}' for model '{model_name}' due to ValueError: {ve}")
            except Exception as e:
                print(f"Unexpected error for user '{username}' and model '{model_name}': {e}")

        return user_results, user_recommendations

    # Parallelize user processing
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_user, username): username for username in users_to_test}

        for future in concurrent.futures.as_completed(futures):
            username = futures[future]
            try:
                user_results, user_recommendations = future.result()
                if user_results:
                    results.extend(user_results)
                if user_recommendations:
                    recommendations[username] = user_recommendations
            except Exception as e:
                print(f"Error processing user {username}: {e}")

    metrics_df = pd.DataFrame(results)
    return metrics_df, recommendations

async def main(users_to_test, df, get_user_data, threshold_movie_list, data_file=None):
    """
    Main function to fetch data, save/load it, and evaluate models.

    Parameters:
        users_to_test (list): List of usernames to evaluate.
        df (pd.DataFrame): Ratings DataFrame.
        get_user_data (function): Asynchronous function to fetch user data.
        threshold_movie_list (list): List of movies to recommend from.
        data_file (str): Path to save/load user data.

    Returns:
        metrics_df (pd.DataFrame): DataFrame containing metrics for each model and user.
        recommendations (dict): Nested dictionary with recommendations for each user and model.
    """
    # Check if data_file exists
    if data_file:
        try:
            # Try loading user data from the file
            user_data_dict = load_user_data(data_file)
        except FileNotFoundError:
            # If file not found, fetch and save data
            print(f"{data_file} not found. Fetching user data.")
            user_data_dict = await fetch_all_user_data(users_to_test, get_user_data)
            save_user_data(user_data_dict, data_file)
    else:
        # If no file provided, fetch user data
        user_data_dict = await fetch_all_user_data(users_to_test, get_user_data)

    print(f"Fetched data for {len(user_data_dict)} users.")

    # Evaluate models using the fetched data
    metrics_df, recommendations = evaluate_models(users_to_test, user_data_dict, df, threshold_movie_list)
    return metrics_df, recommendations

# List of users you want to evaluate
usernames = pd.read_csv('letterboxd_users.csv')
users_to_test = list(usernames['username'])

# File to save/load user data
data_file = 'user_data_test.pkl'

# Run the main function
metrics_df, recommendations = asyncio.run(main(users_to_test, df, get_user_data, threshold_movie_list, data_file=data_file))


# In[ ]:


summary = metrics_df.groupby('model').mean(numeric_only=True)
print(summary)


# In[ ]:


#with open("recommendations.pkl", "wb") as file:
    #pickle.dump(recommendations, file)


# In[ ]:


#metrics_df.to_csv('metrics_results_ranked.csv')


# In[ ]:




