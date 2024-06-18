# ML_Practice
A repository containing the notebooks of some ML related projects I am working on

## [Recommendation/](/Recommendation/)
This repository contains a Recommender System built on the concepts of collaborative filtering. The system can be used to provide personalized recommendations based on user ratings and preferences. It Uses Nearest Neighbors algorithm to train recommendation models and is designed to handle datasets with specific structures and can be customized for different types of items (e.g., movies, books, etc.). 

#### Usage
The main class for the Recommender System is RecommenderSystem. It can be initialized and used as follows:
```
from recommender_system import RecommenderSystem

# Initialize the Recommender System
recommender_system = RecommenderSystem(ratings_df, items_df, user_ratings_filepath, user_id, item_min=50, user_min=200, item_identifier=["title", "author"], rating_mid_point=6, n_neighbors_train=10)

# Clean up datasets
recommender_system.clean_up_datasets()

# Train models
recommender_system.train_models_and_pivot_df()

# Get recommendations
recommendations = recommender_system.pool_and_recommend(max_recommendations=10)
print("Recommendations:", recommendations)
```

#### Dataset Structure
For the Recommender System to work effectively, ensure your datasets meet the following conditions:

The system requires two datasets:
- Ratings Dataset: Should contain columns userId, itemId, and rating.
- Items Dataset: Should include column itemId and at least columns included in `item_identifier` kwarg (the `item_identifier` is a list of columns that uniquely identify an item, eg [title, author] or [title, director].)

Example structure for the ratings dataset:
```
userId,itemId,rating
1,1,5
2,1,3
...
```

Example structure for the items dataset:
```
itemId,title,author
1,The Great Gatsby,F. Scott Fitzgerald
2,To Kill a Mockingbird,Harper Lee
...
```

### [Streamlit Recommendation/](/Recommendation/streamlit_recommendation)
An application of the Recommender system on a movies dataset. Streamlit was used to build a user-friendly interface. To get it to run 

1. Clone the repository
```
git clone https://github.com/The-Professor99/ML_Practice.git
cd Recommendation/streamlit_recommendation
```

2. Create and activate a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages
```
pip install -r requirements.txt
```

4. To run the Streamlit app
```
streamlit run home_page.py
```
