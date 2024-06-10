# the contents in this file has not been modified for performance or readability.
# There are loads of repeating for loops, remove them
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st

class RecommenderSystem():
    def __init__(self, ratings_df, items_df, user_ratings_filepath, user_id, item_min=50, user_min=200, item_identifier=["title"], rating_mid_point=6, n_neighbors_train=10):
        self.df = pd.DataFrame([], columns=['userId', 'itemId', 'rating', 'title'])
        self.ratings_df = ratings_df
        self.items_df = items_df
        self.item_min = item_min
        self.user_min = user_min
        self.item_identifier = item_identifier
        self.ratings_filepath = user_ratings_filepath
        self.user_id = user_id
        self.rating_mid_point = rating_mid_point
        self.n_neighbors_train = n_neighbors_train
        
        # To:Do: in future, make it possible to pass item and user models already trained outside the class
        self.item_model = None
        self.user_model = None
        self.item_df_pivot = None

    def clean_up_datasets(self):
        """
        Clean and merge the ratings and items datasets.
    
        This function performs the following operations:
        1. Remove rows with missing values from both datasets.
        2. Normalize the 'itemId' and 'userId' columns in the ratings dataset.
        3. Normalize the specified item identifier columns in the items dataset.
        4. Merge the datasets on the 'itemId' column.
        5. Retain users with at least the specified minimum number of ratings.
        6. Retain items with at least the specified minimum number of ratings.
        7. Remove duplicate entries based on 'userId' and the item identifier columns.
    
        Parameters:
        df_ratings (DataFrame): The ratings dataset. Columns should be ["userId", "itemId", "rating"]
        df_items (DataFrame): The items dataset. Columns should be ["itemId", ...item_identifier]
        item_min (int): Minimum number of ratings required for an item to be retained.
        user_min (int): Minimum number of ratings required for a user to be retained.
        item_identifier (list): List of column names that uniquely identify an item.
    
        Returns:
        DataFrame: The cleaned and merged dataset.
        """
    
        # Drop rows with missing values
        df_items_clean = self.items_df.copy().dropna()
        df_ratings_clean = self.ratings_df.copy().dropna()
    
        # Normalize columns in ratings dataset
        df_ratings_clean["itemId"] = df_ratings_clean["itemId"].astype(str).str.lower().str.strip()
        df_ratings_clean["userId"] = df_ratings_clean["userId"].astype(str).str.lower().str.strip()
    
        # Normalize columns in items dataset
        df_items_clean["itemId"] = df_items_clean["itemId"].astype(str).str.lower().str.strip()
        for item in self.item_identifier:
            df_items_clean[item] = df_items_clean[item].astype(str).str.lower().str.strip()
    
        # Merge datasets on 'itemId' column
        merged_df = df_ratings_clean.merge(df_items_clean, on="itemId")
    
        # Get occurrencies of each user. How often they appear = how often they rated.
        user_counts = merged_df.groupby('userId').size()
    
        # Get occurrencies of each item. How often they appear = how often they're rated.
        # "title" and "author" are used since two items may share the same title and an author
        # may have multiple items. The combination of title and author points to a unique item.
        item_counts = merged_df.groupby(self.item_identifier
                                  ).size().reset_index(name='counts')
    
        # Choose users to retain (users with at least 200 ratings)
        users_to_keep = user_counts[user_counts >= self.user_min].index
    
        # Choose items to retain (items with atleast 50 ratings)
        items_to_keep = item_counts[item_counts["counts"] >= self.item_min]
    
        # Retain items_to_keep through merge
        merged_df = merged_df.merge(items_to_keep, on=self.item_identifier, how="inner")
    
        # Retain users_to_keep through merge
        merged_df = merged_df[merged_df['userId'].isin(users_to_keep)]
    
        # Drop the 'counts' column and remove duplicates. Each user(given by userId) and item(given by item_identifier) combination should be unique
        merged_df.drop(columns=['counts'], inplace=True)
        merged_df.drop_duplicates(subset=["userId"] + self.item_identifier, inplace=True)
    
        self.df = merged_df
   
    def get_samples(self):
        """Get a list of random samples from the dataset"""
        return self._get_df()[self.item_identifier].sample(5).values
    
    def get_user_edge_rated_books(self, user, edge="top"):
        # print(user, "user", self._get_df_user())
        queried_data =  self._get_df_user().query(f'''userId == "{user}"''')
        if edge == "top":
            # top-most rated first
            queried_data = queried_data[queried_data["rating"] >= self.rating_mid_point].sort_values(by="rating", ascending=False)
        else:
            # lowest ratings first
            queried_data = queried_data[queried_data["rating"] < self.rating_mid_point].sort_values(by="rating")
        return queried_data[self.item_identifier].values

    def retrieve_item_id(self, df, item):
        item_occurrences = df[df[self.item_identifier].isin(item).all(axis=1)]
        if item_occurrences.empty:
            return None
        else:
            return item_occurrences["itemId"].iloc[0]

    # make static method
    def retrieve_user_ratings(self):
        cols = ['itemId', 'rating', 'userId'] + self.item_identifier
        try:
            user_ratings = pd.read_pickle(self.ratings_filepath)
        except FileNotFoundError:
            user_ratings = pd.DataFrame(columns=cols)
        return user_ratings
    
    def update_my_ratings(self, item, rating=None):
        print(item, rating, "--")
        # allows rating same item twice, entry of same item with highest rating will be choosen
        item_display_string = " - ".join(item)
        item_id = self.retrieve_item_id(self._get_df(), item) 
        if item_id:
            user_ratings = self.retrieve_user_ratings() 
            if not rating:
                rating = collect_book_rating(item_display_string)
            rating_to_add = pd.DataFrame([[item_id, rating, self.user_id] + item], columns=user_ratings.columns)
            
            if user_ratings.empty:
                user_ratings = rating_to_add
            else:
                user_ratings = pd.concat([user_ratings, rating_to_add], ignore_index=True)
            user_ratings.to_pickle(self.ratings_filepath) 
            print("Your ratings dataset has been updated!")
        else:
            raise IndexError(f"{item_display_string} not found. Ensure you've entered correct values present in dataset!")

    def train_models_and_pivot_df(self):
        item_model, item_df_pivot = train_model(self._get_df(), self.item_identifier, ["userId"], self.n_neighbors_train)
        user_model, _ = train_model(self._get_df_user(), ["userId"], self.item_identifier, self.n_neighbors_train)
        self.item_model = item_model
        self.user_model = user_model
        
        # explain why we return item_df_pivot
        # item_df_pivot is returned to be used for predictions. Data to be recommended on must
        # be part of item_df_pivot. if new data, retrain. This is different in the case of user_model
        # whose df_pivot should be new whenever a user rates a book. The dynamic is, user is rating multiple
        # books, hence users similar to it will change as it rates, but the rating a book gets is constant. To:Do
        # come explain this better
        self.item_df_pivot = item_df_pivot

        print("Train success")


    def get_similar_users(self, user_ratings):
        user_id =user_ratings["userId"].unique()[0]
        
        df_concat = pd.concat([self._get_df(), user_ratings])
        df_pivot = get_pivot_table(df_concat, ["userId"], self.item_identifier)

        user_model = self._get_user_model()
        
        similar_users = self.get_recommends(user_model, user_id, df_pivot, seek="userId")

        # similar users is of the form [[user id, proximity],...]
        similar_users_ = np.array(similar_users[1])[:, 0].tolist()
        # To:Check this addition of me as a similar user
        similar_users_.append(similar_users[0])
        
        return similar_users_



    def get_my_recommendations(self):
        item_model = self._get_item_model()
        
        user_ratings = self.retrieve_user_ratings()
        if user_ratings.empty:
            print("You have not rated anything yet. when you rate, improve recommendations blah blah blah. Recommending blindly")
            samples_books = self.get_samples()
        else:
            similar_users = self.get_similar_users(user_ratings)
    
            samples = []
            for user in similar_users:
                
                #get top rated books
                top_rated_books = self.get_user_edge_rated_books(user).tolist()
                samples.append(top_rated_books)
    
            # append random sample so it always generates something new. The going to a restaurant and trying things example
            samples.append(self.get_samples().tolist())
    
            samples_books = flatten(samples)
    
            # sort based on book occurrence. the more a book occurs, the better its recommendations
            samples_books = pd.DataFrame(sorted(samples_books, key=lambda book: sort_key(samples_books, book), reverse=True)).drop_duplicates().values
    
        recommended_books = []
        for sample_book in samples_books[:]:
            similar_books = self.get_recommends(item_model, sample_book, self.item_df_pivot)

            if len(similar_books):
                # vet_recommendation(sample_book, similar_books[0])
                if type(similar_books[0]) == list:
                    recommended_books.append([list(similar_books[0])]) # add the sample book too.
                else:
                    recommended_books.append([similar_books[0]]) # add the sample book too.
                    
                recommended_books.append(np.array(similar_books[1])[:, 0].tolist())
                
        return flatten(recommended_books)[:]


    
    def get_eligible(self, recommendations):
        already_seen = self.retrieve_user_ratings()
        eligible = []
        for recommended in recommendations:
            retrieved_id = self.retrieve_item_id(already_seen, recommended)
            if not retrieved_id:
                # means the item has not been seen by user
                eligible.append(recommended)
        return eligible

    def get_negatified_recommendations(self):
        item_model = self._get_item_model()
        # get my down rated books
        down_rated_books = self.get_user_edge_rated_books(self.user_id, edge="down").tolist()

        
        not_recommended_books = []
        for sample_book in down_rated_books[:]:
            similar_books = self.get_recommends(item_model, sample_book, self.item_df_pivot)
            
            if len(similar_books):
                not_recommended_books.append(np.array(similar_books[1])[:, 0].tolist())
    
        return flatten(not_recommended_books)

    def pool_and_recommend(self, max_recommendations=10):
        my_recommendations = self.get_my_recommendations()
    
        # get books similar to books the user doesn't like
        negative_recommendations = self.get_negatified_recommendations()

        sorted_array = pd.DataFrame(sorted(my_recommendations, key=lambda book: sort_key(my_recommendations, book, negate_list=negative_recommendations), reverse=True)).drop_duplicates().values.tolist()
        eligibles = self.get_eligible(sorted_array)
        print("Your Recommendations")
    
        for index, book in enumerate(eligibles[:max_recommendations]):
            print(index + 1, book)

        return eligibles[:max_recommendations]

    def get_recommends(self, knn_model, sample_name, df_pivot, n_neighbors=5, seek="both"): 
        # function to return recommended books - this will be tested
        
        recommended_books = []
        try:
            if len(sample_name) == 1:
                sample = df_pivot.loc[sample_name].values
            else:
                sample = df_pivot.loc[[sample_name]].values
        except KeyError:
            print(f"{sample_name} entered not found in dataset")
            return recommended_books
        else:
            # if use_mean:
            #     sample_mean = sample[sample != 0].mean()
            #     sample.append(sample_mean)
            n_neighbors += 1
            distances, indices = knn_model.kneighbors(sample, n_neighbors=n_neighbors)
            if seek == "both":
                similar_books = df_pivot.index[indices[0]]
            else:
                similar_books = df_pivot.index[indices[0]].get_level_values(seek)
            recommended_books.append(similar_books[0])
    
            similar_books_distances = []
            for similar_book, distance in zip(similar_books[1:], distances[0][1:]):
                similar_book = similar_book
                # TO:DO check here
                if seek == "both" and (len(self.item_identifier) > 1):
                    book_distance = [similar_book, [distance, 1 - distance]] # if both, similar book will be [book, author], 
                    # make the second part a list of [distance, 1-distance] to maintain homogeneity
                else: 
                    book_distance = [similar_book, distance] 
                similar_books_distances.append(book_distance)
    
            recommended_books.append(similar_books_distances)
    
        return recommended_books
    
    def _get_df(self):
        if self.df.empty:
            self.clean_up_datasets()
        return self.df

    def _get_user_model(self):
        if not self.user_model:
            raise Exception("user model not trained blah blah train and run again")
        return self.user_model    
        
    def _get_item_model(self):
        if not self.item_model:
            raise Exception("item model not trained blah blah train and run again")
        return self.item_model

    def _get_df_user(self):
        user_ratings = self.retrieve_user_ratings()
        if user_ratings.empty:
            return self.df
        else:
            return pd.concat([self.df, user_ratings])
    

def collect_book_rating(item_display_string):
    while True:
        try: 
            rating = float(input(f"Enter rating for {item_display_string}: "))
            if rating > 0 and rating <= 10:
                return rating
        except ValueError:
            pass 
        print("\n ==> Please enter a value between 1 and 10 <== \n")

def get_pivot_table(df, index, columns):
    # create a pivot table. This is similar to TF-IDF process. see https://www.geeksforgeeks.org/recommendation-system-in-python/
    df = df.copy()
    #To:do year was int and my rating had it as float, had issues, check remove this for loop with a better implementation
    for i in index:
        df[i] = df[i].str.lower().str.strip()
    for j in columns:
        df[j] = df[j].str.lower().str.strip()
        
    df_pivot = df.pivot_table(index=index, columns=columns, \
                          values='rating', fill_value=0, aggfunc="max")

    # get mean rating of each row
    df_pivot["mean"] = df_pivot.apply(lambda df: df[df != 0].mean(), axis=1)

    df_pivot.dropna(inplace=True)
    return df_pivot



def train_model(df, index, columns, n_neighbors, metric='cosine', algorithm='brute'):
    df_pivot = get_pivot_table(df, index, columns)

    # convert pivot table to sparse matrix
    df_pivot_sparse = csr_matrix(df_pivot)

    neigh_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
    neigh_model.fit(df_pivot_sparse)

    return neigh_model, df_pivot


def flatten(array):
    return_array = []
    for sub_array in array:
        if type(sub_array) != list:
            return_array.append(sub_array)
        else:
            for item in sub_array:
                return_array.append(item)
    return return_array

def sort_key(item_list, item, negate_list=[]):
    """sort by title and author, items occurring more than once are highly favoured, followed by books written by the same author.
    
    Params.
    ----
    item_list: [[book_name, author_name], ...]
    item: [book_name, author_name]
    """
    item_count = item_list.count(item)
    item_secondary_identifier_count = 0
    if type(item) == list and len(item) > 1:
        item_secondary_identifier = item[1]
        secondary_identifiers = list(np.array(item_list)[:, 1]) 
        item_secondary_identifier_count = secondary_identifiers.count(item_secondary_identifier)
    negate_count = 0
    negate_secondary_identifier_count = 0
    
    # negate_list is list of books that should be least_favoured
    if item in negate_list:
        negate_count = negate_list.count(item)
        if type(item) == list and len(item) > 1:
            secondary_identifiers = list(np.array(negate_list)[:, 1]) # extract author names
            negate_secondary_identifier_count = secondary_identifiers.count(item_secondary_identifier) 
        
    overall_count = item_count - negate_count
    overall_secondary_identifier_count = item_secondary_identifier_count - negate_secondary_identifier_count

    overall =  overall_count**2 + overall_secondary_identifier_count

    return overall


def vet_recommendation(id, recommendation_zero):
    if set(id) != set(recommendation_zero):
        print(f"Please check model, {recommendation_zero} not {id}")


