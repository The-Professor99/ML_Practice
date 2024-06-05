# the contents in this file has not been modified for performance or readability.
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def retrieve_isbn(df, title, author):
    book_occurrences = df[df[["title", "author"]].isin([title, author]).all(axis=1)]
    if book_occurrences.empty:
        return None
    else:
        return book_occurrences["isbn"].iloc[0]
        
def collect_book_rating(title, author):
    active = True
    while active:
        try: 
            rating = int(input(f"Enter rating for {title} by {author}: "))
            if rating < 1 or rating > 10:
                raise ValueError
        except ValueError:
            print("\n ==> Please enter a value between 1 and 10 <== \n")
        else:
            active = False
    return rating
    
def retrieve_my_ratings():
    try:
        my_ratings = joblib.load(ratings_filepath)
    except FileNotFoundError:
        my_ratings = pd.DataFrame(columns=['isbn', 'rating', 'title', 'author', 'user'])
    return my_ratings

def update_my_ratings(title, author):
    # allows rating same book twice, entry of same book with highest rating will be choosen
    book_isbn = retrieve_isbn(df, title, author) 
    if book_isbn:
        my_ratings = retrieve_my_ratings() 
        rating = collect_book_rating(title, author)
        rating_to_add = pd.DataFrame([[book_isbn, float(rating), title, author, my_id]], columns=my_ratings.columns)
        if my_ratings.empty:
            my_ratings = rating_to_add
        else:
            my_ratings = pd.concat([my_ratings, rating_to_add], ignore_index=True)
        joblib.dump(my_ratings, ratings_filepath) 
        print("Your ratings dataset has been updated!")
    else:
        raise IndexError("Book not found. Ensure you've entered correct values!")
    return my_ratings

# After inspecting the datasets
def clean_up_datasets(df_ratings, df_books):
    """
    Clean and merge the ratings and books datasets.

    This function performs the following operations:
    1. Remove rows with missing values from both datasets.
    2. Normalize the 'isbn' and 'user' columns in the ratings dataset.
    3. Normalize the 'isbn' and 'author' columns in the books dataset.
    4. Merge the datasets on the 'isbn' column.
    5. Retain users with at least 200 ratings.
    6. Retain books with at least 50 ratings.
    7. Remove duplicate entries based on 'user', 'title', and 'author'.

    Parameters:
    df_ratings (DataFrame): The ratings dataset.
    df_books (DataFrame): The books dataset.

    Returns:
    DataFrame: The cleaned and merged dataset.
    """

    # Drop rows with missing values
    df_books_clean = df_books.copy().dropna()
    df_ratings_clean = df_ratings.copy().dropna()

    # Normalize columns in ratings dataset
    df_ratings_clean["isbn"] = df_ratings_clean["isbn"].str.lower().str.strip()
    df_ratings_clean["user"] = df_ratings_clean["user"].astype(str).str.lower().str.strip()

    # Normalize columns in books dataset
    df_books_clean["isbn"] = df_books_clean["isbn"].str.lower().str.strip()
    df_books_clean["author"] = df_books_clean["author"].str.lower().str.strip()

    # Merge datasets on 'isbn' column
    merged_df = df_ratings_clean.merge(df_books_clean, on="isbn")

    # Get occurrencies of each user. How often they appear = how often they rated.
    user_counts = merged_df.groupby('user').size()

    # Get occurrencies of each book. How often they appear = how often they're rated.
    # "title" and "author" are used since two books may share the same title and an author
    # may have multiple books. The combination of title and author points to a unique book.
    book_counts = merged_df.groupby(["title", "author"]
                              ).size().reset_index(name='counts')

    # Choose users to retain (users with at least 200 ratings)
    users_to_keep = user_counts[user_counts >= 200].index

    # Choose books to retain (books with atleast 50 ratings)
    books_to_keep = book_counts[book_counts["counts"] >= 50]

    # Retain books_to_keep through merge
    merged_df = merged_df.merge(books_to_keep, on=["title", "author"], how="inner")

    # Retain users_to_keep through merge
    merged_df = merged_df[merged_df['user'].isin(users_to_keep)]

    # Drop the 'counts' column and remove duplicates
    merged_df.drop(columns=['counts'], inplace=True)
    merged_df = merged_df.drop_duplicates(subset=['user', 'title', 'author'])

    return merged_df

def get_df_user():
    my_ratings = retrieve_my_ratings()
    if my_ratings.empty:
        return df
    else:
        return pd.concat([df, my_ratings])

def get_pivot_table(df, index, columns):
    # create a pivot table. This is similar to TF-IDF process. see https://www.geeksforgeeks.org/recommendation-system-in-python/
    df_pivot = df.pivot_table(index=index, columns=columns, \
                          values='rating', fill_value=0, aggfunc="max")

    # get mean rating of each row
    df_pivot["mean"] = df_pivot.apply(lambda df: df[df != 0].mean(), axis=1)

    df_pivot.dropna(inplace=True)
    return df_pivot
    
def train_model(df, index=['title', "author"], columns=['user'], n_neighbors=5, metric='cosine', algorithm='brute'):
    df_pivot = get_pivot_table(df, index, columns)

    # convert pivot table to sparse matrix
    df_pivot_sparse = csr_matrix(df_pivot)

    neigh_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
    neigh_model.fit(df_pivot_sparse)

    return neigh_model, df_pivot

# function to return recommended books - this will be tested
def get_recommends(knn_model, sample_name, df_pivot, n_neighbors=5, seek="both"): 
    recommended_books = []
    try:
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
            if seek == "both":
                book_distance = [similar_book, [distance, 1 - distance]] # if both, similar book will be [book, author], 
                # make the second part a list of [distance, 1-distance] to maintain homogeneity
            else: 
                book_distance = [similar_book, distance] 
            similar_books_distances.append(book_distance)

        recommended_books.append(similar_books_distances)
    return recommended_books

def get_samples():
    """Return a list of samples"""

    # use df cause df_books removes books due to certain criteria not being met
    return df[["title", "author"]].sample(5).values

def vet_recommendation(id, recommendation_zero):
    if set(id) != set(recommendation_zero):
        print("Please check model, recommendation zero not id")

def get_similar_users(my_ratings):

            
    user_id = my_ratings["user"].unique()[0]
    
    df_concat = pd.concat([df, my_ratings])
    df_pivot = get_pivot_table(df_concat, ["user"], ["title", "author"])

    
    similar_users = get_recommends(user_model, user_id, df_pivot, seek="user")
    vet_recommendation(user_id, similar_users[0])

    # similar users is of the form [[user id, proximity],...]
    # this returns user_id only of five similar users
    similar_users = np.array(similar_users[1])[:5, 0]
    
    return similar_users

def get_user_top_rated_books(user):
    return df.query(f'''user == "{user}"''').sort_values(by="rating", ascending=False)[:5][["title", "author"]].values

def flatten(array):
    return_array = []
    for sub_array in array:
        for item in sub_array:
            return_array.append(item)
    return return_array

def sort_key(books_list, book):
    """sort by title and author, books occurring more than once are highly favoured, followed by books written by the same author.
    
    Params.
    ----
    books_list: [[book_name, author_name], ...]
    book: [book_name, author_name]
    """
    # books_list = list(books_list)
    book_author = book[1]
    book_count = books_list.count(book)
    author_names = list(np.array(books_list)[:, 1]) # extract author names
    author_count = author_names.count(book_author)
    return book_count**2 + author_count

def similar_users_recommends(array):
    return pd.Series(sorted(a,key=a.count, reverse=True)).unique()

def get_my_recommendations():
    my_ratings = retrieve_my_ratings()
    if my_ratings.empty:
        print("You have not rated anything yet. when you rate, improve recommendations blah blah blah. Recommending blindly")
        samples_books = get_samples()
    else:
        similar_users = get_similar_users(my_ratings)

        samples = []
        for user in similar_users:
            
            top_rated_books = get_user_top_rated_books(user).tolist()
            samples.append(top_rated_books)

        samples_books = flatten(samples)

        # sort based on book occurrence. the more a book occurs, the better its recommendations
        samples_books = pd.DataFrame(sorted(samples_books, key=lambda book: sort_key(samples_books, book), reverse=True)).drop_duplicates().values

    recommended_books = []
    for sample_book in samples_books[:10]:
        similar_books = get_recommends(book_model, sample_book, books_df)
        
        if len(similar_books):
            vet_recommendation(sample_book, similar_books[0])
            recommended_books.append([list(similar_books[0])]) # add the sample book too.
            recommended_books.append(np.array(similar_books[1])[:5, 0].tolist())
            
    return flatten(recommended_books)[:10]

def get_invalids(recommendations):
    already_read = retrieve_my_ratings()
    invalids = []
    for recommended in recommendations:
        retrieved_isbn = retrieve_isbn(already_read, recommended[0], recommended[1])
        if retrieved_isbn:
            # means the book has already been read
            invalids.append(recommended)
    return invalids

def pool_and_recommend():
    my_recommendations = get_my_recommendations()
    sorted_array = pd.DataFrame(sorted(my_recommendations, key=lambda book: sort_key(my_recommendations, book), reverse=True)).drop_duplicates().values.tolist()
    invalids = get_invalids(sorted_array)
    print("Invalids: ", invalids)
    print("Your Recommendations")

    index = 0
    for book in sorted_array:
        if book not in invalids:
            print(index + 1, book)
            index += 1 