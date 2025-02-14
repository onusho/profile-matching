###############################
# app.py
###############################
from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import _pickle as pickle
from random import sample
from scipy.stats import halfnorm

# -------------------------------
# 1. MongoDB Connection
# -------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]

def load_dataframe(collection_name):
    """Load a DataFrame from a MongoDB collection (removes _id)."""
    cursor = db[collection_name].find()
    records = list(cursor)
    for rec in records:
        rec.pop("_id", None)
    return pd.DataFrame(records)

# Load our preprocessed data from MongoDB:
df = load_dataframe("refined_profiles")            # Original profiles DataFrame
cluster_df = load_dataframe("refined_cluster")       # Cluster information
vect_df = load_dataframe("vectorized_refined")       # Vectorized features DataFrame

with open('.\\data\\combined.pkl', 'rb') as f:
    combined = pickle.load(f)



# Compute min and max age from combined['Age'] (assuming it's a NumPy array)
min_age = 18
max_age = 100

# -------------------------------
# 2. Flask App Configuration
# -------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key_here"  # Needed for flash messages

# -------------------------------
# 3. Load the ML Model (from file)
# -------------------------------
model = load("data/refined_model.joblib")

# -------------------------------
# 4. Helper Functions
# -------------------------------
def string_convert(x):
    """Converts lists to space‐separated strings."""
    if isinstance(x, list):
        return ' '.join(x)
    return x

def vectorization(df, columns, input_df):
    """
    Recursively vectorizes text-based columns using CountVectorizer,
    or encodes categorical columns like Religion/Politics.
    """
    if len(columns) == 0:
        return df, input_df
    
    column_name = columns[0]
    if column_name not in ['Bios', 'Movies', 'Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
        return vectorization(df, columns[1:], input_df)

    if column_name in ['Religion', 'Politics']:
        df[column_name.lower()] = df[column_name].cat.codes
        mapping = {v: k for k, v in dict(enumerate(df[column_name].cat.categories)).items()}
        input_df[column_name.lower()] = mapping[input_df[column_name].iloc[0]]
        df = df.drop(column_name, axis=1)
        input_df = input_df.drop(column_name, axis=1)
        return vectorization(df, df.columns, input_df)
    else:
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(df[column_name].values.astype('U'))
        y = vectorizer.transform(input_df[column_name].values.astype('U'))
        df_words = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
        y_words = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names_out(), index=input_df.index)
        new_df = pd.concat([df, df_words], axis=1).drop(column_name, axis=1)
        y_df = pd.concat([input_df, y_words], axis=1).drop(column_name, axis=1)
        return vectorization(new_df, new_df.columns, y_df)

def scaling(df, input_df):
    """
    Scales the features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(df)
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
    return input_vect

def top_ten(cluster, vect_df, input_vect):
    """
    Finds the top 10 similar profiles (by correlation) within the same cluster.
    """
    # Filter the vectorized DataFrame by cluster
    des_cluster = vect_df[vect_df['cluster'] == cluster[0]].drop(columns=['cluster'])
    # Append new profile
    des_cluster = pd.concat([des_cluster, input_vect], ignore_index=False, sort=False)
    user_n = input_vect.index[0]
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])
    top_10_sim = corr.sort_values(ascending=False)[1:11]  # exclude the new user
    top_10 = df.loc[top_10_sim.index]
    return top_10.astype('object')

def get_example_bios():
    """Return 3 random example bios from df."""
    return sample(list(df['Bios']), 3)

# -------------------------------
# 5. Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # --- Input Validation ---
        bio = request.form.get("bio", "").strip()
        religion = request.form.get("Religion", "")
        politics = request.form.get("Politics", "")
        age = int(request.form.get("Age", ""))
        
        # Check required fields
        if not bio:
            flash("Bio is required.")
            return redirect(url_for("index"))
        if not religion:
            flash("Please select a Religion.")
            return redirect(url_for("index"))
        if not politics:
            flash("Please select a Politics option.")
            return redirect(url_for("index"))
        if not age:
            flash("Age is required.")
            return redirect(url_for("index"))
        
        # Additional fields can be optional or validated similarly.
        # Get other fields (if not provided, empty string is used)
        # For fields with multiple choices, expect a comma-separated string.
        new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1] + 1])
        new_profile['Bios'] = [bio]
        
        # --- Preprocess new_profile for classifier ---
        for c in df.columns:
            df[c] = df[c].apply(string_convert)
            new_profile[c] = new_profile[c].apply(string_convert)
        df_v, input_df = vectorization(df, df.columns, new_profile)
        new_df = scaling(df_v, input_df)
        
        # --- Predict Cluster ---
        cluster = model.predict(new_df)
        
        # --- Fetch same-cluster records & Compute Top 10 ---
        top_10_df = top_ten(cluster, vect_df, new_df)
        
        # --- Add New Profile to MongoDB ---
        doc = new_profile.to_dict(orient='records')[0]
        doc["cluster"] = int(cluster[0])
        db.profiles.insert_one(doc)
        
        # --- Render Results ---
        top10_html = top_10_df.to_html(classes='table table-striped')
        return render_template("results.html", top10=top10_html)
    
    # GET request: Render the form, passing combined options and age range.
    example_bios = get_example_bios()
    return render_template("index.html", example_bios=example_bios, politics=combined['Politics'], religions=combined['Religion'], min_age=min_age, max_age=max_age)

@app.route("/how-it-works")
def how_it_works():
    """Display a page that explains how everything works."""
    return render_template("how_it_works.html")

@app.route("/profiles")
def show_profiles():
    """Display all stored profiles from MongoDB."""
    all_profiles = list(db.refined_profiles.find())
    for profile in all_profiles:
        profile["_id"] = str(profile["_id"])
    return render_template("profiles.html", profiles=all_profiles)

###############################
# 6. Main
###############################
if __name__ == "__main__":
    app.run(debug=True)
