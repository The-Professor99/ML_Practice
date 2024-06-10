import pandas as pd
import re
import streamlit as st
import numpy as np
import pandas as pd
import time
import os

from recommendation_system import RecommenderSystem

@st.cache_data
def get_data():
    missing = {'Babylon 5': '1995',
     'Ready Player One': '2018',
     'Hyena Road': '2015',
     'The Adventures of Sherlock Holmes and Doctor Watson': '1980',
     'Nocturnal Animals': '2016',
     'Paterson': '2016',
     'Moonlight': '2016',
     'The OA': '2016',
     'Cosmos': '2019',
     'Maria Bamford: Old Baby': '2017',
     'Death Note: Desu nôto (2006–2007)': '2006',
     'Generation Iron 2': '2017',
     'Black Mirror': '2011'}

    def extract_year(title):
        title = title["title"]
        c = re.search(r"\((\d{4})\)", title)
        if not c:
            a = title.strip()
            try:
                b = missing[a]
            except KeyError:
                b = input(f"Enter movie-{a} year: ")
                missing[a] = b
        else:
            a = title.replace(c.group(0), "").strip()
            b = c.group(1).strip()
        return pd.Series([a, int(b)])
    # loading movie dataset
    movies = pd.read_csv(os.path.join(".", "movies.csv"), usecols=["movieId", "title", "genres"])
    ratings = pd.read_csv(os.path.join(".", "ratings.csv"), usecols=["userId", "movieId", "rating"])
    
    ratings.rename(columns={"movieId": "itemId"}, inplace=True)
    movies.rename(columns={"movieId": "itemId"}, inplace=True)
    
    movies[["title", "year"]] = movies[["title"]].apply(extract_year, axis=1)
    ratings["rating"] = ratings["rating"].div(5).mul(10) 
    return ratings, movies

def initialize():
    if 'recommended_movies' not in st.session_state:
        st.session_state['recommended_movies'] = []
    if "ratings_updated" not in st.session_state:
        st.session_state["ratings_updated"] = False

# initialize app 
initialize()
try:
    ratings_df, items_df = get_data()
except Exception as e:
    st.write("dd")
    st.write(e)
    st.write('ff')
recommender_system = RecommenderSystem(ratings_df, items_df, "my_ratings_stream.pkl","user_id", item_min=30, user_min=50, item_identifier=["title", "year"], rating_mid_point=6)

def button_callback():
    # recommended = [["test", "1"], ["test2", "3"]]
    recommender_system.train_models_and_pivot_df()
    with st.spinner('Getting Recommendations. This may take a while...'):
        recommended = recommender_system.pool_and_recommend(100)
        st.session_state['recommended_movies'] = recommended

@st.experimental_dialog("Rate Movies")
def rate_movies(): 
    recommended_items = st.session_state['recommended_movies']
    df = pd.DataFrame(recommended_items, columns=["title", "year"])
    df["rating"] = None

    st.write("You don't have to rate everything... lorem")
    edited_df = st.data_editor(df, 
        column_config={
            "rating": st.column_config.NumberColumn(
                "Your rating",
                help="How much do you like this item (1-10)?",
                min_value=1,
                max_value=10,
                step=1,
                format="%d ⭐",
            ),
        },
        disabled=["title", "year"])
    
    if st.button("Save Ratings"):
        with st.spinner('Saving ratings...'):
            edited_rows = edited_df.dropna().values.tolist() # keep only edited records
            for edited in edited_rows:
                item = edited[:2]
                rating = edited[2]
                recommender_system.update_my_ratings(item, rating)  
            st.session_state["ratings_updated"] = True
            st.rerun()


st.header("Movie Recommendation", divider=True)

if st.session_state["ratings_updated"]:
    msg = st.toast('Your lorem was saved!', icon='🎉')
    time.sleep(2)
    msg.toast("Get Newer recommendations by running 'Get Recommendations' again!")
    st.session_state["ratings_updated"] = False
    
left_column, right_column = st.columns(2)


with left_column:
    st.markdown('''
    #### Guidelines
    - Click `Get Recommendations` to get recommendations. Each click returns other recommendations.
    - By rating, you get better recommendations. Click the `Update Ratings` button to rate movies.
    ''')
    
    recommendations_button = st.button('Get Recommendations', on_click=button_callback)

    
with right_column:
    recommended_items = st.session_state['recommended_movies']
    
    tile = right_column.container(height=350)
    with tile:
        st.subheader("Recommendations: ")
        st.table(pd.DataFrame(recommended_items, columns=["title", "year"]))
        
    if len(recommended_items):    
        update_button = st.button("Update Ratings")    
        if update_button:
            rate_movies()
    