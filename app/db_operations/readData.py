import pandas as pd
import random
from sqlalchemy import create_engine, text


def read_csv(fpath):
    # dataset is massive, we are going to have to work with smaller amount of rows first
    p = 0.01  # to randomly select 50% of the rows
    spotify_playlist_df = pd.read_csv(fpath,
                                     on_bad_lines='skip', skiprows=lambda i: i > 0 and random.random() > p)
    # cleaning up the columns here
    spotify_playlist_df.columns = spotify_playlist_df.columns.str.replace('"', '')
    spotify_playlist_df.columns = spotify_playlist_df.columns.str.replace('name', '')
    spotify_playlist_df.columns = spotify_playlist_df.columns.str.replace(' ', '')
    return spotify_playlist_df


def read_db(limit):
    try:
        # Create SQLAlchemy engine
        engine = create_engine("mysql+mysqlconnector://root:@localhost/spotify_tracks")

        # Use the engine to run the SQL query
        with engine.connect() as connection:
            query = f"""SELECT user_id, artist_name, track_name, playlist_name 
                        FROM all_tracks LIMIT {limit}"""
            # Use pd.read_sql to execute the query
            df = pd.read_sql(text(query), connection)
        return df
    except Exception as err:
        print(f"Error during connection or query execution: {err}")


def main():
    try:
        print(read_db(100))
    except Exception as e:
        print(f"Error connecting to the db: {e}")


if __name__ == '__main__':
    main()
