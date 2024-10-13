import mysql.connector
from readData import read_csv

def setup_database(mydb):
    try:
        db_cursor = mydb.cursor()

        # Drop the table if it already exists
        db_cursor.execute("DROP TABLE IF EXISTS all_tracks")

        # Create a new table with all columns as VARCHAR(255) and user_id as PRIMARY KEY (nullable)
        create_table_query = '''
        CREATE TABLE all_tracks (
            user_id VARCHAR(255),
            artist_name VARCHAR(255),
            track_name VARCHAR(255),
            playlist_name VARCHAR(255)
        )
        '''
        db_cursor.execute(create_table_query)
        print("Table 'all_tracks' created successfully.")

    except mysql.connector.Error as err:
        print(f"Error setting up the table: {err}")

    finally:
        db_cursor.close()  # Ensure cursor is closed after setup


def insert_row(mydb, values):
    try:
        db_cursor = mydb.cursor()

        # Define the SQL query with placeholders
        query = '''
            INSERT INTO all_tracks (user_id, artist_name, track_name, playlist_name) 
            VALUES (%s, %s, %s, %s)
        '''

        # Execute the SQL query with the provided values
        db_cursor.execute(query, values)

        # Commit the transaction
        mydb.commit()

        print(f"Record inserted with user_id: {values[0]}")

    except mysql.connector.Error as err:
        print(f"Error loading data into table: {err}")

    finally:
        db_cursor.close()


def main():
    try:
        # Establish a database connection
        mydb = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',  # Add your MySQL password here
            port='3306',
            database='spotify_tracks',
            allow_local_infile=True
        )
        print("Successfully connected to the database.")

        # Set up the table (drop and recreate)
        setup_database(mydb)

        # Path to the CSV file
        fpath = "./data/spotify_dataset.csv"

        # Read the CSV file
        df = read_csv(fpath)

        # Loop through the rows and insert into the new table
        for row in df.itertuples():
            values = (row.user_id, row.artist, row.track, row.playlist)  # Access via dot notation
            insert_row(mydb, values)

    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")

    finally:
        # Close the connection after all rows have been inserted
        if mydb.is_connected():
            mydb.close()
            print("Database connection closed.")


if __name__ == '__main__':
    main()
