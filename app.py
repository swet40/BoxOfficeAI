import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_or_create_model():
    try:
        # Try to load the existing model
        model = joblib.load('box_office_predictor.joblib')
        st.success(" Loaded trained model!")
        return model
    except Exception as e:
        st.warning(" Creating a fallback model...")
        
        # Create a simple fallback model
        X_dummy = pd.DataFrame({
            'budget_log': [18, 16, 19, 17, 15, 20],
            'popularity_log': [5, 4, 6, 4.5, 3.5, 5.5],
            'vote_average': [7.5, 6.0, 8.0, 6.5, 5.5, 7.0],
            'vote_count': [5000, 2000, 8000, 3000, 1000, 6000],
            'has_collection': [1, 0, 1, 0, 0, 1],
            'num_genres': [2, 1, 3, 2, 1, 2]
        })
        y_dummy = [1, 0, 1, 0, 0, 1]  # 1=Hit, 0=Flop
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_dummy, y_dummy)
        
        # Save it for next time
        joblib.dump(model, 'box_office_predictor.joblib')
        
        st.info(" Using fallback model (retrain in Colab for better accuracy)")
        return model

@st.cache_resource
def load_features():
    try:
        with open('model_features.json', 'r') as f:
            return json.load(f)
    except:
        return ['budget_log', 'popularity_log', 'vote_average', 'vote_count', 'has_collection', 'num_genres']

# Load model and features
model = load_or_create_model()
expected_features = load_features()

def create_movie_features(budget, popularity, rating, votes, is_franchise, genres_list):
    """
    Convert user inputs into the format expected by the model
    """
    # Start with all zeros for expected features
    movie_features = {feature: 0 for feature in expected_features}
    
    # Set basic features (these should match your training data)
    movie_features.update({
        'budget_log': np.log1p(budget),
        'popularity_log': np.log1p(popularity),
        'vote_average': rating,
        'vote_count': votes,
        'has_collection': 1 if is_franchise else 0,
        'num_genres': len(genres_list),
    })
    
    # Set genre flags - match the exact genre column names from your training
    for genre in genres_list:
        # Create the genre column name (adjust based on your actual column names)
        genre_col = f"genre_{genre.lower().replace(' ', '_')}"
        
        # Check if this genre column exists in expected features
        if genre_col in movie_features:
            movie_features[genre_col] = 1
        else:
            # If the exact column doesn't exist, try to find a close match
            matching_cols = [col for col in expected_features if genre.lower() in col.lower()]
            if matching_cols:
                movie_features[matching_cols[0]] = 1

    # Convert to DataFrame with the exact column order used during training
    features_df = pd.DataFrame([movie_features])[expected_features]
    
    return features_df

# Streamlit app
# st.set_page_config(page_title="Box Office AI", page_icon="🎬", layout="wide")

# Rest of your app code continues here...
st.title("🎬 Box Office Success Predictor")
st.markdown("Predict if a movie will be a **HIT** or **FLOP** using AI!")

# Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Movie Details")
    budget = st.slider("Production Budget ($)", 1000000, 300000000, 50000000, 1000000)
    popularity = st.slider("Popularity Score", 10, 1000, 200, 10)
    rating = st.slider("Expected Rating (0-10)", 0.0, 10.0, 6.5, 0.1)
    votes = st.slider("Expected Votes", 100, 50000, 5000, 100)
    franchise = st.radio("Part of a Franchise?", ["No", "Yes"])
    
with col2:
    st.subheader("Genres")
    genres = st.multiselect(
        "Select Genres",
        ["Action", "Adventure", "Comedy", "Drama", "Horror", 
        "Sci-Fi", "Thriller", "Romance", "Fantasy", "Animation",
        "Mystery", "Crime", "Family", "Western"]
    )

# Prediction button
if st.button(" Predict Box Office Success", type="primary"):
    if not genres:
        st.error("Please select at least one genre!")
    else:
        # Create features and predict
        movie_df = create_movie_features(
            budget=budget,
            popularity=popularity,
            rating=rating,
            votes=votes,
            is_franchise=(franchise == "Yes"),
            genres_list=genres
        )
        
        prediction = model.predict(movie_df)[0]
        probability = model.predict_proba(movie_df)[0]
        confidence = probability[1] if prediction == 1 else probability[0]
        
        # Display results with graduated messages
        st.subheader("🎭 Prediction Results")
        
        if prediction == 1:  # HIT
            if confidence >= 0.85:
                st.success(f"##  MEGA BLOCKBUSTER! ({confidence:.1%} confidence)")
                st.write("Exceptional hit potential!")
            elif confidence >= 0.70:
                st.success(f"##  STRONG HIT! ({confidence:.1%} confidence)")
                st.write("High chances of box office success!")
            elif confidence >= 0.60:
                st.info(f"## 👍 LIKELY HIT ({confidence:.1%} confidence)")
                st.write("Good potential, but some risk factors")
            else:
                st.warning(f"## 🤔 BORDERLINE HIT ({confidence:.1%} confidence)")
                st.write("Could go either way - marketing will be key")
                
        else:  # FLOP
            if confidence >= 0.85:
                st.error(f"## 💸 BOX OFFICE DISASTER ({confidence:.1%} confidence)")
                st.write("High risk of significant financial loss")
            elif confidence >= 0.70:
                st.error(f"##  LIKELY FLOP ({confidence:.1%} confidence)")
                st.write("Poor prospects for box office success")
            elif confidence >= 0.60:
                st.warning(f"##  BORDERLINE FLOP ({confidence:.1%} confidence)")
                st.write("Might break even with strong marketing")
            else:
                st.info(f"##  TOO CLOSE TO CALL ({confidence:.1%} confidence)")
                st.write("Very uncertain - audience reception will decide")
        
        # Probability breakdown
        st.subheader(" Probability Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Flop Probability", f"{probability[0]:.1%}")
        with col2:
            st.metric("Hit Probability", f"{probability[1]:.1%}")
        
        # Movie summary
        st.subheader("🎥 Movie Summary")
        st.write(f"**Budget:** ${budget:,}")
        st.write(f"**Genres:** {', '.join(genres)}")
        st.write(f"**Franchise:** {franchise}")
        st.write(f"**Popularity:** {popularity}")
        st.write(f"**Expected Rating:** {rating}/10")

# Footer
st.markdown("---")
st.caption("Built with Python, Scikit-learn, and Streamlit | Box Office AI Project")