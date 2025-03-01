import pandas as pd

if __name__ == "__main__":
    # Load artist mapping
    artist_mapping_path = "artist_mapping/artist_mapping_2.dat"
    artist_mapping_df = pd.read_csv(artist_mapping_path, sep="\t", encoding="latin1")

    # Print structure of artist mapping
    print(artist_mapping_df.head())

    # Check if some known artists exist in the mapping
    test_artists = ["Death", "In Flames", "Children of Bodom", "Blind Guardian", "Nightwish"]
    for artist in test_artists:
        print(artist_mapping_df[artist_mapping_df["lastfm_artist_name"] == artist])
