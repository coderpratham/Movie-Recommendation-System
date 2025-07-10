# Content-Based Movie Recommendation System

This project implements a **Content-Based Recommendation System** using metadata from movies such as **genres**, **cast**, **crew**, and **plot overviews** to suggest similar movies. The system is designed to provide personalized movie recommendations based on the content a user already likes or is viewing.

## 📌 Project Overview

### 🔍 Objective
To recommend movies that are **similar in content** to a selected movie using:
- **Text feature extraction (TF-IDF)**
- **Cosine similarity**
- **Content attributes**: overview, genre, top cast, and director

## 📁 Dataset

Datasets from the [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) are used:

- `movies_metadata.csv`
- `credits.csv`

Make sure both are placed in the same directory as the script.

## 🛠️ Features Used

- `overview` – Plot summary of the movie  
- `genres` – Movie genres (Action, Comedy, etc.)  
- `cast` – Top 3 cast members  
- `crew` – Director of the movie  

All of these are combined into a single string (called `soup`) for vectorization.

## 📦 Dependencies
Install the required Python packages using pip:
```bash
pip install pandas scikit-learn
