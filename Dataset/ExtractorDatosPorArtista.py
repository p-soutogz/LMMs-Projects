import requests
import pandas as pd
import time
import os
from tqdm import tqdm

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Obtener todos los contentIds de un artista
def get_paintings_content_ids(artist_slug):
    paintings = []
    url = "https://www.wikiart.org/en/App/Painting/PaintingsByArtist"
    params = {"artistUrl": artist_slug, "json": 2}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        for painting in response.json():
            cid = painting.get('contentId')
            if cid:
                paintings.append(cid)
    else:
        print(f"⚠️ Error {response.status_code} al acceder a pinturas de {artist_slug}")
    return paintings

# Obtener detalles de una pintura
def get_painting_details(content_id):
    url = f"https://www.wikiart.org/en/App/Painting/ImageJson/{content_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            data = response.json()
        except Exception as e:
            print(f"❌ Error parseando JSON para {content_id}: {e}")
            return None
        if data:
            return {
                "title": data.get("title", "No title"),
                "artist_name": data.get("artistName", "Unknown"),
                "style": data.get("style", "Unknown"),
                "genre": data.get("genre", "Unknown"),
                "description": (data.get("description") or "").strip()
            }
    return None

# Descargar todas las pinturas con descripción de un artista
def get_paintings_with_description(artist_slug):
    paintings_with_desc = []
    content_ids = get_paintings_content_ids(artist_slug)
    for cid in tqdm(content_ids, desc=f"Procesando {artist_slug}", leave=False):
        painting = get_painting_details(cid)
        if painting and painting['description']:
            paintings_with_desc.append(painting)
        time.sleep(0.01)
    return paintings_with_desc

# --- CONFIGURACIÓN ---
input_csv = "AllArtistNamesWithSlug.csv"
csv_filename = "all_artists_paintings2.csv"
log_file = "processed_artists.txt"

# Crear CSV si no existe
if not os.path.exists(csv_filename):
    df_empty = pd.DataFrame(columns=["title", "artist_name", "style", "genre", "description"])
    df_empty.to_csv(csv_filename, index=False, encoding='utf-8')

# Leer lista de artistas
df = pd.read_csv(input_csv)
artists = df[["artist_name", "artist_slug"]].dropna()

# Leer artistas ya procesados
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        already_processed = set(line.strip() for line in f)
else:
    already_processed = set()

total_guardadas = 0

# --- PROCESAMIENTO ---
for _, row in artists.iterrows():
    name = row["artist_name"]
    slug = row["artist_slug"]

    if slug in already_processed:
        print(f"⏩ {name} ({slug}) ya procesado. Saltando.")
        continue

    print(f"\n🎨 Descargando obras de {name} ({slug})...")
    paintings = get_paintings_with_description(slug)

    if paintings:
        df_artist = pd.DataFrame(paintings)
        df_artist.to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8')
        total_guardadas += len(df_artist)
        print(f"✅ Guardadas {len(df_artist)} obras de {name}")
    else:
        print(f"⚠️ Ninguna obra válida encontrada para {name}")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{slug}\n")

print(f"\n✅ Proceso finalizado. Se han guardado {total_guardadas} obras en '{csv_filename}'")
