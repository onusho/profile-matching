from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import _pickle as pickle
from random import sample
from scipy.stats import halfnorm



app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'  # Needed for sessions if you expand functionality



# -------------------------------
# Loading your data and models
# -------------------------------

with open("..\\data\\refined_profiles.pkl", 'rb') as fp:
    df = pickle.load(fp)
    
with open("..\\data\\refined_cluster.pkl", 'rb') as fp:
    cluster_df = pickle.load(fp)
    
with open("..\\data\\vectorized_refined.pkl", 'rb') as fp:
    vect_df = pickle.load(fp)
    
model = load("..\\data\\refined_model.joblib")

# -------------------------------
# Helper Functions
# -------------------------------

def string_convert(x):
    """Converts lists to space‐separated strings."""
    if isinstance(x, list):
        return ' '.join(x)
    return x

def vectorization(df, columns, input_df):
    """
    Recursively vectorizes text-based columns using CountVectorizer.
    """
    column_name = columns[0]
    if column_name not in ['Bios', 'Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
        return df, input_df

    if column_name in ['Religion', 'Politics']:
        df[column_name.lower()] = df[column_name].cat.codes
        # Create a mapping from label to code
        mapping = {v: k for k, v in dict(enumerate(df[column_name].cat.categories)).items()}
        input_df[column_name.lower()] = mapping[input_df[column_name].iloc[0]]
        input_df = input_df.drop(column_name, axis=1)
        df = df.drop(column_name, axis=1)
        return vectorization(df, df.columns, input_df)
    else:
        vectorizer = CountVectorizer()
        # Fit on original data and transform both dataframes
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
    # filtering clustered data and appending new profile
    des_cluster = vect_df[vect_df['cluster'] == cluster[0]].drop(columns=['cluster'])
    des_cluster = pd.concat([des_cluster, input_vect], ignore_index=False, sort=False)
    
    # finding top 10 similar, correlated users in the cluster
    user_n = input_vect.index[0]
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])
    top_10_sim = corr.sort_values(ascending=False)[1:11]        # excluding the user
    
    top_10 = df.loc[top_10_sim.index]
    return top_10.astype('object')
    

def get_example_bios():
    examples = []
    for i in sample(list(df.index), 3):
        examples.append(df['Bios'].loc[i])
    return examples

# -------------------------------
# Data for Input Choices
# -------------------------------

with open('..\\data\\categories.pkl', 'rb') as f:
    combined = pickle.load(f)
with open('..\\data\\probability.pkl', 'rb') as f:
    p = pickle.load(f)
# -------------------------------
# Routes
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve form inputs
        bio = request.form.get("bio")
        random_vals = request.form.get("random_vals") == "on"
        
        # Create a new profile DataFrame
        new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1] + 1])
        new_profile['Bios'] = [bio]
        
        # Loop over other columns (skipping Bios)
        for col in new_profile.columns[1:]:
            if random_vals:
                if col in ['Religion', 'Politics']:
                    new_profile[col] = np.random.choice(combined[col], 1, p=p[col])
                elif col == 'Age':
                    new_profile[col] = int(halfnorm.rvs(loc=18, scale=8, size=1))
                else:
                    choices = list(np.random.choice(combined[col], size=3, p=p[col]))
                    new_profile[col] = [list(set(choices))]
            else:
                if col in ['Religion', 'Politics']:
                    new_profile[col] = request.form.get(col)
                elif col == 'Age':
                    new_profile[col] = request.form.get(col)
                else:
                    # For list fields, we expect a comma-separated string
                    val = request.form.get(col)
                    if val:
                        selected = [x.strip() for x in val.split(",") if x.strip()]
                        new_profile[col] = [list(set(selected))]
                    else:
                        new_profile[col] = [[]]
        
        # Convert all columns to proper string format for vectorization
        for col in df.columns:
            df[col] = df[col].apply(string_convert)
            new_profile[col] = new_profile[col].apply(string_convert)
        
        # Process new profile: vectorize, scale, and predict cluster
        df_v, input_df = vectorization(df, df.columns, new_profile)
        new_df = scaling(df_v, input_df)
        cluster = model.predict(new_df)
        top_10_df = top_ten(cluster, vect_df, new_df)
        
        # Convert the DataFrame to HTML to render in the template
        top10_html = top_10_df.to_html(classes='table table-striped')
        return render_template("results.html", top10=top10_html)
    
    # GET request: Render the form
    example_bios = get_example_bios()
    return render_template("index.html", example_bios=example_bios, combined=combined)

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

if __name__ == "__main__":
    app.run(debug=True)
