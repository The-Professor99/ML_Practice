# the contents in this file has not been modified for performance or readability.
# There are loads of repeating for loops, remove them
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st

class RecommenderSystem():
    def __init__(self, ratings_df, items_df, user_ratings_filepath, user_id, item_min=50, user_min=200, item_identifier=["title"], rating_mid_point=6, max_rating=10, n_neighbors_train=10):
        """
        Parameters
        ---
        ratings_df: The ratings dataset. should contain columns `userId`, `itemId` and `rating` showing users ratings on different items.
        items_df: The items dataset. Should contain columns `itemId` and columns present in `item_identifier`.
        user_ratings_filepath: path to save and retrieve user ratings. a pickle file path is expected. If not existent, will be created.
        user_id: a unique identifier used in saving user's ratings to df.
        item_min: Minimum number of ratings required for an item to be included.
        user_min: Minimum number of ratings required for a user to be included.
        item_identifier: List of columns that uniquely identify an item.
        rating_mid_point: Midpoint value to distinguish between high and low ratings.
        max_rating: max rating an item can have.
        n_neighbors_train: Number of neighbors to consider during model training.
        """
        self.df = pd.DataFrame([], columns=['userId', 'itemId', 'rating', 'title'])
        self.ratings_df = ratings_df
        self.items_df = items_df
        self.item_min = item_min
        self.user_min = user_min
        self.item_identifier = item_identifier
        self.ratings_filepath = user_ratings_filepath
        self.user_id = user_id.lower()
        self.rating_mid_point = rating_mid_point
        self.n_neighbors_train = n_neighbors_train
        self.max_rating = max_rating
        
        # To:Do: in future, make it possible to pass item and user models already trained outside the class
        self.item_model = None
        self.user_model = None
        self.item_df_pivot = None

    def clean_up_datasets(self):
        """
        Clean and merge the ratings and items datasets. 
        """
        df_items_clean = self.items_df.copy().dropna()
        df_ratings_clean = self.ratings_df.copy().dropna()
    
        df_ratings_clean["itemId"] = df_ratings_clean["itemId"].astype(str).str.lower().str.strip()
        df_ratings_clean["userId"] = df_ratings_clean["userId"].astype(str).str.lower().str.strip()
    
        df_items_clean["itemId"] = df_items_clean["itemId"].astype(str).str.lower().str.strip()
        for item in self.item_identifier:
            df_items_clean[item] = df_items_clean[item].astype(str).str.lower().str.strip()
    
        merged_df = df_ratings_clean.merge(df_items_clean, on="itemId")
    
        user_counts = merged_df.groupby('userId').size()
    
        item_counts = merged_df.groupby(self.item_identifier
                                  ).size().reset_index(name='counts')
    
        users_to_keep = user_counts[user_counts >= self.user_min].index
    
        items_to_keep = item_counts[item_counts["counts"] >= self.item_min]
    
        merged_df = merged_df.merge(items_to_keep, on=self.item_identifier, how="inner")
    
        merged_df = merged_df[merged_df['userId'].isin(users_to_keep)]
    
        merged_df.drop(columns=['counts'], inplace=True)
        merged_df.drop_duplicates(subset=["userId"] + self.item_identifier, inplace=True)
    
        self.df = merged_df
   
    def get_samples(self, num_samples=5):
        """Get a list of random samples from the dataset"""
        return self._get_df()[self.item_identifier].sample(num_samples).values
    
    def get_user_edge_rated_items(self, user_id, edge="top"): 
        """
        Retrieve the user's highest or lowest rated items.
    
        Parameters:
        ----------
        user_id : str
            The user ID for which the rated items are to be retrieved.
        edge : str, optional
            Specifies whether to retrieve the top-rated or lowest-rated items. 
            Acceptable values are "top" for top-most rated items and "down" for 
            lowest-rated items. The default is "top".
        """        
        queried_data =  self._get_df_users().query(f'''userId == "{user_id}"''')
        if edge == "top":
            queried_data = queried_data[queried_data["rating"] >= self.rating_mid_point].sort_values(by="rating", ascending=False)
        else:
            queried_data = queried_data[queried_data["rating"] < self.rating_mid_point].sort_values(by="rating")
        return queried_data

    def sample_items(self, df, max_items=50, edge="top"):
        """
        Sample items from the DataFrame with a weighted probability based on ratings.
    
        Parameters:
        df (pd.DataFrame): The DataFrame containing items and their ratings.
        max_items (int): The maximum number of items to sample. 
        edge (str): Determines the weighting strategy. if "top", items with 
                    high ratings are given higher weights, if "down", items with
                    lower ratings are given higher weights. 
    
        Returns:
        np.ndarray: Array of sampled item identifiers.
        """
        num_items = min(max_items, len(df))
        weights = df["rating"].pow(2) if edge == "top" else df["rating"].rsub(self.max_rating).pow(2)
        if weights.empty:
            weights = None
    
        return df.sample(num_items, weights=weights)[self.item_identifier].values
    
    def retrieve_item_id(self, df, item):
        """
        Retrieve the item ID for a given item from the DataFrame.
    
        Parameters:
        df (pd.DataFrame): The DataFrame containing items and their identifiers.
        item (list): The item attributes to match in the DataFrame.
    
        Returns:
        int or None: The item ID if found, otherwise None.
        """
        item_occurrences = df[df[self.item_identifier].isin(item).all(axis=1)]
        return item_occurrences["itemId"].iloc[0] if not item_occurrences.empty else None

    def retrieve_ratings_in_file(self):    
        """
        Retrieve ratings in the file.
    
        Returns:
        pd.DataFrame: A DataFrame containing the ratings. If the file does not exist, 
        returns an empty DataFrame with predefined columns.
        """
        cols = ['itemId', 'rating', 'userId'] + self.item_identifier
        try:
            ratings_in_file = pd.read_pickle(self.ratings_filepath)
        except FileNotFoundError:
            ratings_in_file = pd.DataFrame(columns=cols)
        return ratings_in_file

    def retrieve_user_ratings(self):    
        """
        Retrieve the ratings for the current user.
    
        """
        return self.retrieve_ratings_in_file().query(f"userId == '{self.user_id}'")
    
    def update_users_ratings(self, item, rating=None):
        """
            Update the user's ratings dataset with a new rating for a specified item.
        
        
            Parameters:
            ----------
            item : list
                The item to be rated, identified by its attributes. 
            rating : float, optional
                The rating to be assigned to the item. If not provided, the function 
                will prompt the user to input a rating.
        
            Raises:
            ------
            IndexError
                If the specified item is not found in the dataset.
        """
        item_display_string = " - ".join(item)
        item_id = self.retrieve_item_id(self._get_df(), item) 
        if item_id:
            ratings_in_file = self.retrieve_ratings_in_file() 
            rating = check_rating(rating, self.max_rating) or  collect_item_rating(item_display_string, self.max_rating)
            rating_to_add = pd.DataFrame([[item_id, rating, self.user_id] + item], columns=ratings_in_file.columns)
            ratings_in_file = pd.concat([ratings_in_file, rating_to_add], ignore_index=True) if not ratings_in_file.empty else rating_to_add
            ratings_in_file.to_pickle(self.ratings_filepath) 
            print("Your ratings dataset has been updated!")
        else:
            raise IndexError(f"{item_display_string} not found. Ensure you've entered correct values present in dataset!")

    def train_models_and_pivot_df(self):
        """
        Train the item and user models and create the item pivot table.
        """
        self.item_model, self.item_df_pivot = train_model(self._get_df(), self.item_identifier, ["userId"], self.n_neighbors_train)
        self.user_model, _ = train_model(self._get_df(), ["userId"], self.item_identifier, self.n_neighbors_train)

        print("Train success")


    def get_similar_users(self, user_ratings):
        """
        Retrieve a list of users similar to the given user based on their ratings
        """
        user_id = user_ratings["userId"].unique()[0]
        df_concat = pd.concat([self._get_df(), user_ratings])
        df_pivot = get_pivot_table(df_concat, ["userId"], self.item_identifier)
        similar_users = self.generate_similar_items(self._get_user_model(), user_id, df_pivot, seek="userId")
        return np.array(similar_users[1])[:, 0].tolist()
        
    def get_similar_items(self, item_list):
        """
        Generate similar items for each item in the input item list.
        
        Returns:
        -------
        list
            A flattened list containing each item from the input item list along with their similar items
        """
        similar_items_container = []
        for item in item_list:
            similar_items = self.generate_similar_items(self._get_item_model(), item, self.item_df_pivot)
            if len(similar_items):
                similar_items_container.append([list(item)]) if isinstance(similar_items[0], tuple) else similar_items_container.append([item[0]])
                similar_items_container.append(np.array(similar_items[1])[:, 0].tolist())
        return flatten(similar_items_container)

    def generate_similar_items(self, model, item, df_pivot, n_neighbors=5, seek="both"): 
        """
        Generate items similar to a given item using the trained model.
        
        Parameters:
        - model: The trained model for generating similar items.
        - item (str): The item to find similar items for.
        - df_pivot (pd.DataFrame): The pivot table used for similarity calculation.
        - seek (str): Columns to identify the item in the pivot table.
        
        Returns:
        -------
        list
            A list containing similar items based on the trained model.
        """        
        similar_items_container = []
        try:
            item_vector  = df_pivot.loc[item].values if len(item) == 1 else df_pivot.loc[[item]].values
        except KeyError:
            print(f"{item} entered not found in dataset")
            return similar_items_container
        else:
            n_neighbors += 1
            distances, indices = model.kneighbors(item_vector, n_neighbors=n_neighbors)
            similar_items = df_pivot.index[indices[0]] if seek == "both" else df_pivot.index[indices[0]].get_level_values(seek)
            similar_items_container.append(similar_items[0])
            similar_items_distances = []
            for similar_item, distance in zip(similar_items[1:], distances[0][1:]):
                item_distance = [similar_item, [distance, 1 - distance]] if seek == "both" and (len(self.item_identifier) > 1) else [similar_item, distance] 
                similar_items_distances.append(item_distance)
            similar_items_container.append(similar_items_distances)
        return similar_items_container
        
    def get_positive_recommendations(self):
        """
        Generates personalized item recommendations for the user based on 
        their ratings and the ratings of similar users(collaborative filtering). If the user has not rated any items, 
        random samples are used for recommendations.
    
        Returns:
        -------
        list
            A list of recommended items for the user.
        """
        user_ratings = self.retrieve_user_ratings()
        if user_ratings.empty:
            print("You have not rated anything yet. When you rate, your recommendations will improve. Recommending randomly...")
            initial_items  = self.get_samples()
        else:
            similar_users = self.get_similar_users(user_ratings)
            similar_users.append(self.user_id) 
            samples = []
            for user in similar_users: 
                top_rated_items = self.get_user_edge_rated_items(user)
                sampled_high_ratings = self.sample_items(top_rated_items).tolist()
                samples.append(sampled_high_ratings)
            initial_items = flatten(samples)
            initial_items  = pd.DataFrame(sorted(initial_items, key=lambda item: sort_key(initial_items , item), reverse=True)).drop_duplicates().values

        # We could return the items rated highly by similar users(samples_items) but I decided to, in addition to these items, find items similar to them and include them as recommendations also. To:Do: Decide if this is good logic
        recommended_items = self.get_similar_items(initial_items)
                
        return recommended_items
    
    def get_eligible(self, recommendations):
        """
        Filter out recommendations that the user has already seen or rated.
    
        Parameters:
        -----------
        recommendations : list
            List of recommended items to filter.
    
        Returns:
        --------
        list
            List of items that the user has not seen or rated yet (eligible recommendations).
        """
        already_seen = self.retrieve_user_ratings()
        eligible = []
        for recommended in recommendations:
            retrieved_id = self.retrieve_item_id(already_seen, recommended)
            if not retrieved_id:
                eligible.append(recommended)
                
        return eligible

    def get_negative_recommendations(self):    
        """
        Generate a list of recommendations based on items the user has rated negatively (down-rated).
    
        This function retrieves items that the user has rated poorly and uses them to generate a list
        of items similar to the poorly rated ones.
    
        Returns:
        -------
        list
            A list of items based the user may rate negatively.
        """
        down_rated_items = self.get_user_edge_rated_items(self.user_id, edge="down")
        sampled_low_ratings  = self.sample_items(down_rated_items, edge="down").tolist()
        negative_recommendations = self.get_similar_items(sampled_low_ratings )
        return negative_recommendations

    def generate_user_recommendations(self, max_recommendations=10):
        """
        Generate personalized item recommendations for the user based on collaborative filtering.
        
        This function generates recommendations tailored to the user's preferences by leveraging collaborative filtering
        techniques. It considers the user's positive ratings and penalizes items similar to those the user has rated negatively.
        
        Returns:
        -------
        list
            A list of recommended items for the user.
        """
        positive_recommendations = self.get_positive_recommendations()
        negative_recommendations = self.get_negative_recommendations()
        sorted_recommendations = pd.DataFrame(sorted(positive_recommendations, key=lambda item: sort_key(positive_recommendations, item, negate_list=negative_recommendations), reverse=True)).drop_duplicates().values.tolist()
        eligibles = self.get_eligible(sorted_recommendations)
        return eligibles[:max_recommendations]

    def _get_df(self):
        if self.df.empty:
            self.clean_up_datasets()
        return self.df

    def _get_user_model(self):
        if not self.user_model:
            self.train_models_and_pivot_df()
        return self.user_model    
        
    def _get_item_model(self):
        if not self.item_model:
            self.train_models_and_pivot_df()
        return self.item_model

    def _get_df_users(self):
        user_ratings = self.retrieve_user_ratings()
        if user_ratings.empty:
            return self._get_df()
        else:
            return pd.concat([self._get_df(), user_ratings])


def check_rating(rating, max_rating):
    """ 
    Validate the provided rating.
    
    Parameters:
    - rating (float): The rating to be validated.
    - max_rating (float): The maximum value a rating can be..
    
    Returns:
    - float or None: The validated rating or None if the rating is invalid.
   """
    if rating > 0 and rating <= max_rating:
        return rating
    else:
        print(f"\n ==>  Rating should be between {0} and {max_rating}(inclusive) <== \n")
        return None
    
def collect_item_rating(item_display_string, max_rating):
    """
    Collect a rating from the user for a specified item.
    """
    while True:
        try: 
            rating = check_rating(float(input(f"Enter rating for {item_display_string}: ")), max_rating)
            if rating:
                return rating
        except ValueError:
            pass 

def get_pivot_table(df, index, columns):
    """
    Create a pivot table from the input DataFrame based on specified index and columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing user-item ratings.
    index : list
        List of column names to be used as index in the pivot table.
    columns : list
        List of column names to be used as columns in the pivot table.

    Returns:
    --------
    pandas.DataFrame
        Pivot table with index as 'index', columns as 'columns', and values as 'rating'.
        It also includes a 'mean' column representing the mean rating for each row.
    """
    # This is similar to TF-IDF process. see https://www.geeksforgeeks.org/recommendation-system-in-python/
    
    for i in index:
        df[i] = df[i].str.lower().str.strip()
    for j in columns:
        df[j] = df[j].str.lower().str.strip()
        
    df_pivot = df.pivot_table(index=index, columns=columns, \
                          values='rating', fill_value=0, aggfunc="max")
    df_pivot["mean"] = df_pivot.apply(lambda df: df[df != 0].mean(), axis=1)
    df_pivot.dropna(inplace=True)
    return df_pivot


def train_model(df, index, columns, n_neighbors, metric='cosine', algorithm='brute'):
    """
    Train a nearest neighbors model for recommendation based on user-item ratings.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing user-item ratings.
    index : list
        List of column names to be used as index in the pivot table.
    columns : list
        List of column names to be used as columns in the pivot table.
    n_neighbors : int
        Number of neighbors to consider for the nearest neighbors model.
    metric : str, optional
        Distance metric to use for calculating similarity ('cosine' by default).
    algorithm : str, optional
        Algorithm to use for nearest neighbors computation ('brute' by default).

    Returns:
    --------
    sklearn.neighbors.NearestNeighbors
        Trained nearest neighbors model.
    pandas.DataFrame
        Pivot table of user-item ratings.
    """
    
    df_pivot = get_pivot_table(df, index, columns)
    df_pivot_sparse = csr_matrix(df_pivot)
    neigh_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
    neigh_model.fit(df_pivot_sparse)
    return neigh_model, df_pivot

def flatten(array):
    """
    Flatten a nested list structure into a single list.
    """
    flattened = []    
    for item in array:
        if isinstance(item, list):
            flattened.extend(item)  
        else:
            flattened.append(item)
    return flattened

def sort_key(item_list, item, negate_list=[]):
    """
    Calculate a sorting key for an item based on its occurrence in item_list and negate_list.
    
    Parameters:
    -----------
    item_list : list
        List of items to calculate sorting score against.
    item : list
        Item [item_name, author_name] for which the sorting score is calculated.
    negate_list : list, optional
        List of items to be negatively weighted in sorting.

    Returns:
    --------
    float
        Sorting key value based on item's occurrence and its secondary identifier count.

    Notes:
    ------
    - Items occurring more than once are favored.
    - items with the same author as `item` are also favored.
    - `negate_list` contains items that should be least favored in sorting.
    
     Examples:
    ---------
    # item_identifier == ["title", "author"]
    >>> item_list = [['item1', 'Author1'], ['item2', 'Author2'], ['item1', 'Author1']]
    >>> item = ['item1', 'Author1']
    >>> negate_list = [['item2', 'Author2']]
    >>> sort_key(item_list, item, negate_list)
    
    """
    def calculate_total(main_id_pos_count, main_id_neg_count, sec_id_pos_count, sec_id_neg_count):
        return ((main_id_pos_count - main_id_neg_count) ** 2) + (sec_id_pos_count - sec_id_neg_count)

    item_count = item_list.count(item)
    item_secondary_identifier_count = 0
    negate_count = 0
    negate_secondary_identifier_count = 0
    if isinstance(item, list) and len(item) > 1:
        item_secondary_identifier = item[1]
        secondary_identifiers = list(np.array(item_list)[:, 1]) 
        item_secondary_identifier_count = secondary_identifiers.count(item_secondary_identifier)
    if item in negate_list:
        negate_count = negate_list.count(item)
        if isinstance(item, list) and len(item) > 1:
            secondary_identifiers = list(np.array(negate_list)[:, 1]) 
            negate_secondary_identifier_count = secondary_identifiers.count(item_secondary_identifier) 
    return calculate_total(item_count, negate_count, item_secondary_identifier_count, negate_secondary_identifier_count)
    