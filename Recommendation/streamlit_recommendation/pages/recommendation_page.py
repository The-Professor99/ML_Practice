import os
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from recommendation_system import RecommenderSystem
from utils import get_file_path

@st.cache_data
def get_data():
    movies = pd.read_csv(get_file_path("movies.csv"), usecols=["itemId", "title", "year"])
    ratings = pd.read_csv(get_file_path("ratings.csv"), usecols=["userId", "itemId", "rating"])

    return ratings, movies

def logout():
    #simulate loading
    time.sleep(2)
    
    # if anonymous user, delete file on logout
    if st.session_state["user_email"] == "Anon":
        rating_file_name = get_file_path(f"{st.session_state["user_id"]}_ratings.pkl")
        if os.path.exists(rating_file_name):
            os.remove(rating_file_name)
        
    st.session_state["user_id"] = None
    st.session_state["user_email"] = None
    st.query_params["anonymous"] = False  
    st.session_state['recommended_movies'] = []
    st.session_state["ratings_updated"] = False
    st.rerun()

def initialize():
    if 'recommended_movies' not in st.session_state:
        st.session_state['recommended_movies'] = []
    if "ratings_updated" not in st.session_state:
        st.session_state["ratings_updated"] = False
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None 
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = None  
    if "anonymous" not in st.query_params:
        st.query_params["anonymous"] = False  

def generate_dummy_user_id():
    rand_nums = "-".join(map(str, np.random.randint(0, 9, 9)))
    time_created = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    uid = f"user_{time_created}-{rand_nums}"
    return uid

def get_recommendations(recommender_system):
    with st.spinner('Getting Recommendations. This may take a while...'):
        recommender_system.train_models_and_pivot_df()
        recommended = recommender_system.pool_and_recommend(100)
        st.session_state['recommended_movies'] = recommended

@st.experimental_dialog("Rate Movies")
def rate_movies(recommender_system): 
    recommended_items = st.session_state['recommended_movies']
    df = pd.DataFrame(recommended_items, columns=["title", "year"])
    df["rating"] = None

    st.write("Help us improve your recommendations by rating movies. Rate only the movies you've watched. It's okay to skip those you haven't. ")
    st.write("=== Rating Scale: 1 - 10 ===")
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
            
def main():
    initialize()
    
    #Check if the user is already logged in or is anonymous user
    if (not st.session_state["user_id"]) and (st.query_params["anonymous"] != "True"):
        # route to login page
        st.switch_page("pages/login_page.py")

    # if user logged in or is anonymous, get user_id from session state
    if st.session_state["user_id"]:
        user_id = st.session_state["user_id"] 
    elif st.query_params["anonymous"]:
        user_id =  generate_dummy_user_id() 
        st.session_state["user_id"] = user_id
        st.session_state["user_email"] = "Anon"

    # also get training data and initialize recommender system
    ratings_df, items_df = get_data()
    user_ratings_file = f"{user_id}_ratings.pkl"
    recommender_system = RecommenderSystem(ratings_df, items_df, get_file_path(user_ratings_file), user_id, item_min=30, user_min=50, item_identifier=["title", "year"], rating_mid_point=6)

    # Display user email and logout button on sidebar
    with st.sidebar:
        st.text(st.session_state["user_email"])
        logout_clicked = st.button("Logout")
        if logout_clicked:
            with st.spinner("Logging you out..."):
                logout()  

    # == Start: Main content Display ==
    
    st.header("Movie Recommendation", divider=True)
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown('''
        #### Guidelines
        1. **Get Recommendations**: Click the "Get Recommendations" button to receive a list of movie recommendations tailored to your tastes. Each click returns newer recommendations.
        2. **Rate Movies**: Help us improve your recommendations by rating the movies you've watched. You can update your ratings at any time. Click the `Update Ratings` button to rate movies.
         ''')
        
        recommendations_button = st.button('Get Recommendations') 
        if recommendations_button:
            get_recommendations(recommender_system)
    with right_column:
        recommended_items = st.session_state['recommended_movies']
        
        tile = right_column.container(height=350)
        with tile:
            st.subheader("Recommendations: ")
            st.table(pd.DataFrame(recommended_items, columns=["title", "year"]))
            
        if len(recommended_items):    
            update_button = st.button("Update Ratings")    
            if update_button:
                rate_movies(recommender_system)

    # Toast display
    if st.session_state["ratings_updated"]:
        msg = st.toast('Your ratings were saved!', icon='🎉')
        time.sleep(2)
        msg.toast("Get Newer recommendations by running 'Get Recommendations' again!")
        st.session_state["ratings_updated"] = False

    # == End: Main Content Display ==

if __name__ == "__main__":
    main()








    

    