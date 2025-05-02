import requests
import pandas as pd

url = "http://www.wikiart.org/en/App/Artist/AlphabetJson"
params = {
    "v": "new",
    "inPublicDomain": "true"
}

headers = {
    "User-Agent": "Mozilla/5.0"
}

print("📥 Descargando lista de nombres y slugs de artistas...")
response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    artist_records = [
        {
            "artist_name": artist.get("artistName"),
            "artist_slug": artist.get("url")
        }
        for artist in data if artist.get("artistName") and artist.get("url")
    ]

    df = pd.DataFrame(artist_records)
    df.drop_duplicates(subset=["artist_slug"], inplace=True)
    df.to_csv("AllArtistNamesWithSlug.csv", index=False)
    print(f"✅ {len(df)} artistas guardados en 'AllArtistNamesWithSlug.csv'")
else:
    print(f"❌ Error en la petición: {response.status_code}")


