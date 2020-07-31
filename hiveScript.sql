
-- create database for recommendation engine
DROP DATABASE IF EXISTS rec_engine;
CREATE DATABASE IF NOT EXISTS rec_engine;

USE rec_engine;

-- create table schema for movies metadata table
DROP TABLE IF EXISTS movies_metadata;
CREATE TABLE IF NOT EXISTS movies_metadata
( adult BOOLEAN, belongs_to_collection STRING, budget INT, genres STRING, homepage STRING, id INT, imdb_id STRING, 
	original_language STRING, original_title STRING, overview STRING, popularity FLOAT, poster_path STRING, 
	production_companies STRING, production_countries STRING, release_date STRING, revenue FLOAT, runtime FLOAT, 
	spoken_languages STRING, status STRING, tagline STRING, title STRING, video BOOLEAN, vote_average FLOAT, vote_count INT);

-- create table schema for credits table
DROP TABLE IF EXISTS credits;
CREATE TABLE IF NOT EXISTS credits ( casts STRING, crew STRING, id STRING);

-- create movies_ratings data table schema. Open CSV SerDe lib for specifying field seperator and skip headers
DROP TABLE IF EXISTS movies_ratings;
CREATE TABLE movies_ratings ( user_id INT, movie_id INT, rating INT, `time_stamp` INT )
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    "field.delim"=",",
  	"skip.header.line.count"="1"
);

-- data should be in local file system /home/hive folder. otherwise can't load.
LOAD DATA LOCAL INPATH 'ratings_small.csv'
OVERWRITE INTO TABLE movies_ratings;

-- calculate values of movie pairs to find pearson correlation coefficient
DROP TABLE IF EXISTS movie_pair_ratings;
CREATE TABLE movie_pair_ratings AS
SELECT
  a.movie_id ,
  b.movie_id AS movie_id_2 ,
  COUNT(*) AS N ,
  SUM(a.rating) AS ratingSum ,
  SUM(b.rating) AS rating2Sum ,
  SUM(a.rating * b.rating) AS dotProductSum ,
  SUM(a.rating * a.rating) AS ratingSqSum ,
  SUM(b.rating * b.rating) AS rating2SqSum  
FROM
  movies_ratings a
  JOIN movies_ratings b ON a.user_id = b.user_id
WHERE
  a.movie_id < b.movie_id
GROUP BY
 a.movie_id, b.movie_id
HAVING
  N >= 30;

-- find pearson correlation coefficient
DROP TABLE IF EXISTS movie_pair_correlations;
CREATE TABLE movie_pair_correlations AS
SELECT
  movie_id AS movie_id,
  movie_id_2 AS movie_id_2,
  (N * dotProductSum - ratingSum * rating2Sum) / (SQRT(N * ratingSqSum - ratingSum * ratingSum) * SQRT(N * rating2SqSum - rating2Sum * rating2Sum))
   AS correlation
FROM
  movie_pair_ratings
ORDER BY
  movie_id, correlation DESC;

-- join with movies metadata table for finding movie title for first movie id
DROP TABLE IF EXISTS movie_pair_correlations_with_movie_id_titles;
CREATE TABLE movie_pair_correlations_with_movie_id_titles AS
SELECT
  movie_pair_correlations.movie_id,
  movies_metadata.title AS movie_id_title,
  movie_pair_correlations.movie_id_2,
  movie_pair_correlations.correlation
FROM
  movie_pair_correlations
JOIN
  movies_metadata
ON
  movie_pair_correlations.movie_id = movies_metadata.id
ORDER BY
  movie_pair_correlations.movie_id, movie_pair_correlations.correlation DESC;

-- join with movies medadata table for finding movie title for second movie id
DROP TABLE IF EXISTS movie_pair_correlations_with_movie_id_1_2_titles;
CREATE TABLE movie_pair_correlations_with_movie_id_1_2_titles AS
SELECT
  movie_pair_correlations_with_movie_id_titles.movie_id,
  movie_pair_correlations_with_movie_id_titles.movie_id_title,
  movie_pair_correlations_with_movie_id_titles.movie_id_2,
  movies_metadata.title AS movie_id_2_title,
  movie_pair_correlations_with_movie_id_titles.correlation
FROM
  movie_pair_correlations_with_movie_id_titles
JOIN
  movies_metadata
ON
  movie_pair_correlations_with_movie_id_titles.movie_id_2 = movies_metadata.id
ORDER BY
  movie_pair_correlations_with_movie_id_titles.movie_id, movie_pair_correlations_with_movie_id_titles.correlation DESC;


