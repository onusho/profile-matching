from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import _pickle as pickle
from random import sample

client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]

with open(".\\data\\refined_profiles.pkl",'rb') as fp:
    df = pickle.load(fp)

with open(".\\data\\refined_cluster.pkl", 'rb') as fp:
    cluster_df = pickle.load(fp)
    
with open(".\\data\\vectorized_refined.pkl", 'rb') as fp:
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


app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"  

def string_convert(x):
    if isinstance(x, list):
        return ' '.join(x)
    return x

def vectorization(df, columns, input_df):

    column_name = columns[0]
        
    if column_name not in ['Bios', 'Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
                
        return df, input_df
    
    if column_name in ['Religion', 'Politics']:        
        df[column_name.lower()] = df[column_name].cat.codes
        d = dict(enumerate(df[column_name].cat.categories))
        d = {v: k for k, v in d.items()}
        input_df[column_name.lower()] = d[input_df[column_name].iloc[0]]
        input_df = input_df.drop(column_name, axis=1)
        df = df.drop(column_name, axis=1)
        
        return vectorization(df, df.columns, input_df)
    
    else:
        vectorizer = CountVectorizer()        
        x = vectorizer.fit_transform(df[column_name].values.astype('U'))
        y = vectorizer.transform(input_df[column_name].values.astype('U'))
        df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
        y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names_out(), index=input_df.index)
        new_df = pd.concat([df, df_wrds], axis=1)
        y_df = pd.concat([input_df, y_wrds], axis=1)
        new_df = new_df.drop(column_name, axis=1)
        y_df = y_df.drop(column_name, axis=1)
        
        return vectorization(new_df, new_df.columns, y_df) 


def scaling(df, input_df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
    return input_vect

def top_ten(cluster, vect_df, input_vect):
    des_cluster = vect_df[vect_df['cluster']==cluster[0]].drop('cluster', axis=1)
    des_cluster = pd.concat([des_cluster, input_vect], ignore_index=False)
    user_n = input_vect.index[0]
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])
    top_10_sim = corr.sort_values(ascending=False)[1:11]
    top_10 = df.loc[top_10_sim.index]
    top_10[top_10.columns[1:]] = top_10[top_10.columns[1:]]
    
    return top_10.astype('object')


def get_example_bios():
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
        
        # flash('Form submitted successfully!', 'success')
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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        new_profile = get_profile()
        for c in df.columns:
            df[c] = df[c].apply(string_convert)
            new_profile[c] = new_profile[c].apply(string_convert)
        df_v, input_df = vectorization(df, df.columns, new_profile)
        new_df = scaling(df_v, input_df)
        cluster = model.predict(new_df)
        top_10_df = top_ten(cluster, vect_df, new_df)
        top10_html = top_10_df.to_html(classes='table table-striped')
        return render_template("results.html", top10=top10_html)
    example_bios = get_example_bios()
    return render_template("index.html", example_bios=example_bios, politics=combined['Politics'], religions=combined['Religion'], min_age=MIN_AGE,  max_age=MAX_AGE)

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/profiles")
def show_profiles():
    all_profiles = list(db.refined_profiles.find())
    for profile in all_profiles:
        profile["_id"] = str(profile["_id"])
    return render_template("profiles.html", profiles=all_profiles)

if __name__ == "__main__":
    app.run(debug=True)
