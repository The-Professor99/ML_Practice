import streamlit as st
from utils import get_file_path

st.header("Welcome to the Movie Recommendation System!", divider=True)
st.markdown('''
This is a simple recommendation system built on the concepts of collaborative filtering.
It is designed to provide you with personalized movie recommendations based on your ratings and preferences. Here’s how it works:

1. **Get Recommendations**: Click the "Get Recommendations" button to receive a list of movie recommendations tailored to your tastes.
2. **Rate Movies**: Help us improve your recommendations by rating the movies you've watched. You can update your ratings at any time.
3. **Login**: Sign in with your email to save your ratings and get even more personalized recommendations. You can also log in anonymously if you prefer.

''')

image_path = get_file_path('recommendation_image.gif')
st.image(image_path, caption='An overview of the recommendation system')

st.write("Enjoy exploring and discovering new movies tailored just for you!")

button_clicked = st.button("Go To Recommendations")
if button_clicked:
    st.switch_page("pages/recommendation_page.py")



