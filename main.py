#!/usr/bin/env python


import os
from sklearn.feature_extraction.text import TfidfVectorizer
from pyhive import hive
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# open connection with hive table
conn = hive.Connection(host="172.18.0.2")

# read data from hive table
dataframe = pd.read_sql("SELECT movies_metadata.title,movies_metadata.overview, movies_metadata.vote_average, movies_metadata.vote_count FROM rec_engine.movies_metadata LIMIT 10000",conn)
#print(dataframe["movies_metadata.belongs_to_collection"][0])


# from hive table C and m
#Cn= dataframe['movies_metadata.vote_average'].mean()
#mn= dataframe['movies_metadata.vote_count'].quantile(0.9)
#print(f"vote_avarage mean [ {Cn} ], vote_count quantile 0.9 [ {mn} ]")


#def weighted_ratingn(xn, m=mn, C=Cn):
#    v = xn['movies_metadata.vote_count']
#    R = xn['movies_metadata.vote_average']
#    # Calculation based on the IMDB formula
#    return (v/(v+m) * R) + (m/(m+v) * C)



# for hive
tfidfn = TfidfVectorizer(stop_words='english')


# for hive
dataframe['movies_metadata.overview'] = dataframe['movies_metadata.overview'].fillna('')


# for hive
tfidf_matrixn = tfidfn.fit_transform(dataframe['movies_metadata.overview'])
print(f"fit_transform{tfidf_matrixn}")


# for hive
cosine_simn = linear_kernel(tfidf_matrixn, tfidf_matrixn)
print(f"similarity metrix [{cosine_simn}]")


# for hive
indicesn = pd.Series(dataframe.index, index=dataframe['movies_metadata.title']).drop_duplicates()
print(f"indices [{indicesn}]")

def get_recommendationsn(title, cosine_sim=cosine_simn):
    if title not in dataframe['movies_metadata.title'].values:
        return ('This movie is not in our hadoop cluster.\nPlease check if you spelled it correct using camel casing')
    else:
        # Get the index of the movie that matches the title
        # if title not in df2
        print(f"{title}\n")
        idx = indicesn[title]
        print(f"{idx}\n")
        # Get the pairwsie similarity scores of all movies with that movie
#        print(cosine_sim[idx])
        
        sim_scores = list(enumerate(cosine_sim[idx]))
 #       print(sim_scores)

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return dataframe['movies_metadata.title'].iloc[movie_indices]


def get_recommendations_based_on_pcc(title):
    dataframe_correlation = pd.read_sql(f"SELECT movie_id_2_title, correlation from rec_engine.movie_pair_correlations_with_movie_id_1_2_titles WHERE movie_id_title = '{title}' ORDER BY correlation DESC LIMIT 10",conn)
    print(dataframe_correlation)
    if dataframe_correlation.empty:
        return('This movie is not in our database')
    else:
        print("data in our system")
        return dataframe_correlation.movie_id_2_title


app = Flask(__name__)

@app.route("/home_old")
def home_old():
    return render_template('home.html')

@app.route("/")
def home():
    return render_template('homep.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = get_recommendationsn(movie)
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')


@app.route("/recommend_with_pcc")
def recommend_with_pcc():
    movie = request.args.get('movie')
    r = get_recommendations_based_on_pcc(movie)
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8243)

