-- in order to use this you need to start grunt shell as -useHcatalog or pass argument -useHCatalog pig view right below argument list.
REGISTER '/usr/lib/pig/lib/piggybank-0.17.0.jar'
movies_metadata  = LOAD '/user/maria_dev/dataset/movies_metadata.csv' USING org.apache.pig.piggybank.storage.CSVExcelStorage
	(',', 'NO_MULTILINE', 'UNIX', 'SKIP_INPUT_HEADER') AS
	( adult: boolean, belongs_to_collection: chararray, budget: int, genres: chararray, homepage: chararray, id: int, imdb_id: chararray, 
	original_language: chararray, original_title: chararray, overview: chararray, popularity: float, poster_path: chararray, 
    production_companies: chararray, production_countries: chararray, release_date: chararray, revenue: float, runtime: float, 
    spoken_languages: chararray, status: chararray, tagline: chararray, title: chararray, video: boolean, vote_average: float, vote_count: int);


STORE movies_metadata_lite INTO 'rec_engine.movies_metadata' USING org.apache.hive.hcatalog.pig.HCatStorer();

credits  = LOAD '/user/maria_dev/dataset/credits.csv' USING org.apache.pig.piggybank.storage.CSVExcelStorage
	(',', 'NO_MULTILINE', 'UNIX', 'SKIP_INPUT_HEADER') AS
	( casts: chararray, crew: chararray, id: chararray);

STORE credits INTO 'rec_engine.credits' USING org.apache.hive.hcatalog.pig.HCatStorer();


