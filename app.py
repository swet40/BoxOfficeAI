import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Set page config first
st.set_page_config(
    page_title="Box Office AI Predictor",
    page_icon="🎬", 
    layout="wide"
)

st.title(" Box Office Success Predictor")
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
        ["Action", "Adventure", "Comedy", "Drama", "Horror", "Sci-Fi", "Thriller", "Romance", "Fantasy"]
    )

# Advanced rule-based prediction (no scikit-learn dependency)
def advanced_rule_based_prediction(budget, popularity, rating, votes, is_franchise, genres_list):
    """
    Sophisticated rule-based prediction that mimics ML behavior
    """
    score = 0
    confidence = 0.5  # Base confidence
    
    # Budget analysis (0-25 points)
    if budget > 200000000:  # Blockbuster budget
        score += 25
        confidence += 0.3
    elif budget > 100000000:  # Big budget
        score += 20
        confidence += 0.2
    elif budget > 50000000:  # Medium budget
        score += 15
        confidence += 0.1
    elif budget > 20000000:  # Small budget
        score += 10
    else:  # Micro budget
        score += 5
        confidence -= 0.1
    
    # Popularity analysis (0-20 points)
    if popularity > 700:  # Viral popularity
        score += 20
        confidence += 0.2
    elif popularity > 400:  # High popularity
        score += 15
        confidence += 0.15
    elif popularity > 200:  # Moderate popularity
        score += 10
        confidence += 0.1
    elif popularity > 100:  # Low popularity
        score += 5
    else:  # Very low popularity
        score += 2
        confidence -= 0.1
    
    # Rating analysis (0-20 points)
    if rating > 8.5:  # Excellent rating
        score += 20
        confidence += 0.25
    elif rating > 7.5:  # Very good rating
        score += 15
        confidence += 0.15
    elif rating > 6.5:  # Good rating
        score += 10
        confidence += 0.1
    elif rating > 5.5:  # Average rating
        score += 5
    else:  # Poor rating
        score += 0
        confidence -= 0.1
    
    # Franchise power (0-15 points)
    if is_franchise:
        score += 15
        confidence += 0.2
    
    # Genre analysis (0-10 points)
    hit_genres = ["Action", "Adventure", "Sci-Fi", "Animation", "Fantasy"]
    niche_genres = ["Horror", "Thriller", "Comedy"]
    risky_genres = ["Drama", "Romance", "Documentary"]
    
    genre_bonus = 0
    for genre in genres_list:
        if genre in hit_genres:
            genre_bonus += 3
        elif genre in niche_genres:
            genre_bonus += 2
        elif genre in risky_genres:
            genre_bonus += 1
    
    score += min(10, genre_bonus)
    confidence += min(0.15, genre_bonus * 0.02)
    
    # Votes analysis (0-10 points)
    if votes > 20000:  # Massive audience
        score += 10
        confidence += 0.1
    elif votes > 10000:  # Large audience
        score += 8
        confidence += 0.08
    elif votes > 5000:  # Good audience
        score += 6
        confidence += 0.05
    elif votes > 2000:  # Moderate audience
        score += 4
    else:  # Small audience
        score += 2
    
    # Calculate final prediction
    max_score = 100
    hit_probability = score / max_score
    
    # Adjust confidence based on score consistency
    if 30 <= score <= 70:  # Borderline cases have lower confidence
        confidence = max(0.5, confidence - 0.2)
    else:  # Clear cases have higher confidence
        confidence = min(0.95, confidence + 0.1)
    
    prediction = 1 if hit_probability > 0.6 else 0  # Hit threshold at 60%
    
    return prediction, hit_probability, confidence, score

# Prediction button
if st.button("🎯 Predict Box Office Success", type="primary"):
    if not genres:
        st.error("Please select at least one genre!")
    else:
        # Use advanced rule-based prediction
        prediction, hit_probability, confidence, score = advanced_rule_based_prediction(
            budget=budget,
            popularity=popularity,
            rating=rating,
            votes=votes,
            is_franchise=(franchise == "Yes"),
            genres_list=genres
        )
        
        # Display results
        st.subheader("🎭 Prediction Results")
        
        if prediction == 1:
            if confidence >= 0.85:
                st.success(f"## 🚀 MEGA BLOCKBUSTER! ({confidence:.1%} confidence)")
                st.write("Exceptional hit potential! This movie has all the makings of a box office sensation.")
            elif confidence >= 0.75:
                st.success(f"## ✅ STRONG HIT! ({confidence:.1%} confidence)")
                st.write("High chances of box office success with strong commercial potential.")
            elif confidence >= 0.65:
                st.info(f"## 👍 LIKELY HIT ({confidence:.1%} confidence)")
                st.write("Good potential for success, though some marketing effort will be needed.")
            else:
                st.warning(f"## 🤔 BORDERLINE HIT ({confidence:.1%} confidence)")
                st.write("Could go either way - strategic marketing and audience reception will be crucial.")
        else:
            if confidence >= 0.85:
                st.error(f"## 💸 BOX OFFICE DISASTER ({confidence:.1%} confidence)")
                st.write("High risk of significant financial loss. Consider revising the strategy.")
            elif confidence >= 0.75:
                st.error(f"## ❌ LIKELY FLOP ({confidence:.1%} confidence)")
                st.write("Poor prospects for box office success based on current parameters.")
            elif confidence >= 0.65:
                st.warning(f"## ⚠️ BORDERLINE FLOP ({confidence:.1%} confidence)")
                st.write("Might break even with exceptional marketing and word-of-mouth.")
            else:
                st.info(f"## 🔄 TOO CLOSE TO CALL ({confidence:.1%} confidence)")
                st.write("Very uncertain - audience reception and marketing will make or break this film.")
        
        # Probability breakdown
        st.subheader("📊 Analysis Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Flop Probability", f"{(1 - hit_probability):.1%}")
        with col2:
            st.metric("Hit Probability", f"{hit_probability:.1%}")
        with col3:
            st.metric("Prediction Score", f"{score}/100")
        
        # Detailed analysis
        with st.expander("🔍 Detailed Analysis"):
            st.write(f"**Budget Impact:** {'High' if budget > 100000000 else 'Medium' if budget > 30000000 else 'Low'}")
            st.write(f"**Popularity Level:** {'Viral' if popularity > 600 else 'High' if popularity > 300 else 'Moderate' if popularity > 150 else 'Low'}")
            st.write(f"**Rating Quality:** {'Excellent' if rating > 8.0 else 'Good' if rating > 7.0 else 'Average' if rating > 6.0 else 'Poor'}")
            st.write(f"**Franchise Power:** {'Yes' if franchise == 'Yes' else 'No'}")
            st.write(f"**Genre Appeal:** {', '.join(genres)}")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction == 0 and budget > 100000000:
                st.warning("Consider reducing budget or strengthening franchise elements")
            if popularity < 150:
                st.info("Boost marketing to increase popularity score")
            if len(genres) == 1 and genres[0] in ["Drama", "Romance"]:
                st.info("Consider adding complementary genres to broaden appeal")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Box Office AI Project - Advanced Rule-Based Analytics")

# Sidebar info
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    This AI uses advanced rule-based analytics to predict movie success based on:
    
    - **Budget analysis** and financial modeling
    - **Popularity metrics** and market trends  
    - **Genre performance** patterns
    - **Franchise power** and brand value
    - **Historical success** factors
    
    *No machine learning model dependency - 100% deployment reliable*
    """)