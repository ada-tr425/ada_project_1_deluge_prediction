from pathlib import Path
import pandas as pd
from flood_tool.geo import get_gps_lat_long_from_easting_northing

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "resources" / "example_data" / "postcodes_missing_data.csv"
DISTRICT = ROOT / "resources" / "preprocessed_data" / "district_imputed.csv"

OUTPUT = ROOT / "resources" / "preprocessed_data" / "postcodes_missing_features_adapted.csv"


def main():
    print("Loading example unlabelled data...")
    df = pd.read_csv(EXAMPLE)

    print("Extracting postcodeDistrict...")
    df["postcodeDistrict"] = df["postcode"].str.split().str[0]

    print("Merging district_imputed.csv...")
    df_dist = pd.read_csv(DISTRICT)
    df = df.merge(df_dist, on="postcodeDistrict", how="left")

    print("Converting easting/northing → longitude/latitude...")
    lat, lon = get_gps_lat_long_from_easting_northing(
        df["easting"].values,
        df["northing"].values,
        dms=False
    )
    df["latitude"] = lat
    df["longitude"] = lon

    print("Filling required missing numeric fields...")
    df["medianPrice"] = df["medianPrice"] if "medianPrice" in df else df["easting"].median()
    df["historicallyFlooded"] = 0
    df["distanceToWatercourse"] = df["distanceToWatercourse"].fillna(0)
    df["elevation"] = df["elevation"].fillna(0)

    print(f"Saving adapted file to: {OUTPUT}")
    df.to_csv(OUTPUT, index=False)



if __name__ == "__main__":
    main()
