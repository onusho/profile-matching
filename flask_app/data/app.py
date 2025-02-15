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

# -------------------------------
# 1. MongoDB Connection
# -------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]

# def load_dataframe(collection_name):
#     """Load a DataFrame from a MongoDB collection (removes _id)."""
    # cursor = db[collection_name].find()
    # records = list(cursor)
    # for rec in records:
    #     rec.pop("_id", None)
    # return pd.DataFrame(records)
with open("refined_profiles.pkl",'rb') as fp:
    df = pickle.load(fp)

with open("refined_cluster.pkl", 'rb') as fp:
    cluster_df = pickle.load(fp)
    
with open("vectorized_refined.pkl", 'rb') as fp:
    vect_df = pickle.load(fp)

# Load our preprocessed data from MongoDB:
# df = load_dataframe("refined_profiles")            # Original profiles DataFrame
# cluster_df = load_dataframe("refined_cluster")       # Cluster information
# vect_df = load_dataframe("vectorized_refined")       # Vectorized features DataFrame

with open('.\\data\\combined.pkl', 'rb') as f:
    combined = pickle.load(f)

with open('.\\data\\vectorizers.pkl', 'rb') as f:
    vectorizers = pickle.load(f)

model = load(".\\data\\refined_model.joblib")
scaler = load(".\\data\\scaler.joblib")

MIN_AGE = 18
MAX_AGE = 100

# -------------------------------
# 2. Flask App Configuration
# -------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"  


# -------------------------------
# 4. Helper Functions
# -------------------------------
def string_convert(x):
    """Converts lists to space‐separated strings."""
    if isinstance(x, list):
        return ' '.join(x)
    return x

def vectorize(df, columns, input_df):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """

    column_name = columns[0]
        
    # Checking if the column name has been removed already
    if column_name not in ['Bios', 'Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
                
        return df, input_df
    
    # Encoding columns with respective values
    if column_name in ['Religion', 'Politics']:
        
        # Getting labels for the original df
        df[column_name.lower()] = df[column_name].cat.codes
        
        # Dictionary for the codes
        d = dict(enumerate(df[column_name].cat.categories))
        
        d = {v: k for k, v in d.items()}
                
        # Getting labels for the input_df
        input_df[column_name.lower()] = d[input_df[column_name].iloc[0]]
                
        # Dropping the column names
        input_df = input_df.drop(column_name, 1)
        
        df = df.drop(column_name, 1)
        
        return vectorize(df, df.columns, input_df)
    
    # Vectorizing the other columns
    else:
        # Instantiating the Vectorizer
        vectorizer = CountVectorizer()
        
        # Fitting the vectorizer to the columns
        x = vectorizer.fit_transform(df[column_name].values.astype('U'))
        
        y = vectorizer.transform(input_df[column_name].values.astype('U'))

        # Creating a new DF that contains the vectorized words
        df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
        
        y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names_out(), index=input_df.index)

        # Concating the words DF with the original DF
        new_df = pd.concat([df, df_wrds], axis=1)
        
        y_df = pd.concat([input_df, y_wrds], axis=1)

        # Dropping the column because it is no longer needed in place of vectorization
        new_df = new_df.drop(column_name, axis=1)
        
        y_df = y_df.drop(column_name, axis=1)
        
        return vectorize(new_df, new_df.columns, y_df) 


def scaling(input_df):
    """
    Scales the features using MinMaxScaler.
    """
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
    return input_vect

def top_ten(cluster, vect_df, input_vect):
    """
    Returns the DataFrame containing the top 10 similar profiles to the new data
    """
    # Filtering out the clustered DF
    des_cluster = vect_df[vect_df['Cluster #']==cluster[0]].drop('Cluster #', 1)
    
    # Appending the new profile data
    des_cluster = des_cluster.append(input_vect, sort=False)
        
    # Finding the Top 10 similar or correlated users to the new user
    user_n = input_vect.index[0]
    
    # Trasnposing the DF so that we are correlating with the index(users) and finding the correlation
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])

    # Creating a DF with the Top 10 most similar profiles
    top_10_sim = corr.sort_values(ascending=False)[1:11]
        
    # The Top Profiles
    top_10 = df.loc[top_10_sim.index]
        
    # Converting the floats to ints
    top_10[top_10.columns[1:]] = top_10[top_10.columns[1:]]
    
    return top_10.astype('object')


def get_example_bios():
    """Return 3 random example bios from df."""
    return sample(list(df['Bios']), 3)

def get_data():
        bio = request.form.get('bio')
        religion = request.form.get('Religion')
        politics = request.form.get('Politics')
        age = int(request.form.get('Age'))
        movies = request.form.get('Movies', '')
        music = request.form.get('Music', '')
        social_media = request.form.get('Social Media', '')
        sports = request.form.get('Sports', '')
        
        if not bio or not religion or not politics or not age:
            flash('Please fill in all required fields.', 'danger')
            return redirect(url_for('index'))
        try:
            age = int(age)
        except ValueError:
            flash('Age must be a valid number.', 'danger')
            return redirect(url_for('index'))
        
        if age < MIN_AGE or age > MAX_AGE:
            flash(f'Age must be between {MIN_AGE} and {MAX_AGE}.', 'danger')
            return redirect(url_for('index'))
        
        def validate_comma_separated(field_value, field_name):
            if field_value:
                # Split and strip whitespace
                items = [item.strip() for item in field_value.split(',') if item.strip()]
                if len(items) > 3:
                    flash(f'Please select up to 3 {field_name}.', 'danger')
                    return False, []
                return True, items
            return True, []


        valid, movies_list = validate_comma_separated(movies, 'movies')
        if not valid:
            return redirect(url_for('index'))
        
        valid, music_list = validate_comma_separated(music, 'music genres')
        if not valid:
            return redirect(url_for('index'))
        
        valid, social_media_list = validate_comma_separated(social_media, 'social media platforms')
        if not valid:
            return redirect(url_for('index'))
        
        valid, sports_list = validate_comma_separated(sports, 'sports')
        if not valid:
            return redirect(url_for('index'))
        
        flash('Form submitted successfully!', 'success')
        return (bio, movies_list, religion, music_list, politics, social_media_list, sports_list, age)
    
def get_profile():
    categories = get_data()
    profile_index = df.index[-1] + 1
    profile = pd.DataFrame(columns=df.columns, index=[profile_index])
    
    for i, name in enumerate(list(df.columns)):
        profile.at[profile_index, name] = categories[i]

    profile['Religion'] = pd.Categorical(profile.Religion, ordered=True,
                                categories=['Catholic',
                                            'Christian',
                                            'Jewish',
                                            'Muslim',
                                            'Hindu',
                                            'Buddhist',
                                            'Spiritual',
                                            'Other',
                                            'Agnostic',
                                            'Atheist'])

    profile['Politics'] = pd.Categorical(profile.Politics, ordered=True,
                                    categories=['Liberal',
                                                'Progressive',
                                                'Centrist',
                                                'Moderate',
                                                'Conservative'])
    return profile

        

    
# -------------------------------
# 5. Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
         
        profile = get_profile()
        # if profile is not None:
        #     top10_html = df[:10].to_html(classes='table table-striped')

        for c in df.columns:
            profile[c] = profile[c].apply(string_convert)
        df_v, vect_profile = vectorize(df, df.columns, profile)
        vect_profile = scaling(vect_profile)
        
        # # --- Predict Cluster ---
        cluster = model.predict(vect_profile)
        
        # # --- Fetch same-cluster records & Compute Top 10 ---
        top_10_df = top_ten(cluster, vect_df, vect_profile)
        
        # # --- Add New Profile to MongoDB ---
        doc = profile.to_dict(orient='records')[0]
        doc["cluster"] = int(cluster[0])
        db.profiles.insert_one(doc)
        
        # --- Render Results ---
        top10_html = top_10_df.to_html(classes='table table-striped')
        return render_template("results.html", top10=top10_html)
    
    # GET request: Render the form, passing combined options and age range.
    example_bios = get_example_bios()
    return render_template("index.html", example_bios=example_bios, politics=combined['Politics'], religions=combined['Religion'], min_age=MIN_AGE,  max_age=MAX_AGE)

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
