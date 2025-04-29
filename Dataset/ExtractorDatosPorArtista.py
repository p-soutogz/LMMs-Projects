import requests
import pandas as pd
import time
from tqdm import tqdm
import os

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Funciones para obtener datos
def get_paintings_content_ids(artist_url):
    paintings = []
    base_url = "https://www.wikiart.org/en/App/Painting/PaintingsByArtist"
    params = {"artistUrl": artist_url, "json": 2}
    response = requests.get(base_url, headers=headers, params=params)

    if response.status_code == 200:
        painting_list = response.json()
        for painting in painting_list:
            content_id = painting.get('contentId')
            if content_id:
                paintings.append(content_id)
    else:
        print(f"Error al acceder a la API de paintings: {response.status_code}")
    return paintings

def get_painting_details(content_id):
    url = f"https://www.wikiart.org/en/App/Painting/ImageJson/{content_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            data = response.json()
        except Exception as e:
            print(f"Error parseando JSON para {content_id}: {e}")
            return None

        if data and isinstance(data, dict):
            title = data.get('title', 'No title')
            description = (data.get('description') or '').strip()
            artist_name = data.get('artistName', 'Unknown')
            style = data.get('style', 'Unknown')
            genre = data.get('genre', 'Unknown')
            return {
                "title": title,
                "artist_name": artist_name,
                "style": style,
                "genre": genre,
                "description": description,
            }
    else:
        print(f"Error al acceder a detalles del contentId {content_id}: {response.status_code}")
        return None

def get_paintings_with_description(artist_url):
    paintings_with_desc = []
    total_con_descripcion = 0
    total_sin_descripcion = 0
    content_ids = get_paintings_content_ids(artist_url)

    for cid in tqdm(content_ids, desc=f"Procesando obras de {artist_url}"):
        painting = get_painting_details(cid)
        if painting:
            if painting['description']:
                paintings_with_desc.append(painting)
                total_con_descripcion += 1
            else:
                total_sin_descripcion += 1
        else:
            total_sin_descripcion += 1
        time.sleep(0.2)
    
    print(f"Resumen de {artist_url}: {total_con_descripcion} guardadas | {total_sin_descripcion} descartadas por falta de descripción")
    return paintings_with_desc

# --- EJECUCIÓN MÚLTIPLE ---
artists = [
    "vincent-van-gogh", "pablo-picasso", "leonardo-da-vinci", "claude-monet", "salvador-dali",
    "rembrandt", "gustav-klimt", "paul-cezanne", "edgar-degas", "pierre-auguste-renoir",
    "frida-kahlo", "henri-matisse", "andy-warhol", "raphael", "caravaggio",
    "joan-miro", "georges-seurat", "paul-gauguin", "jackson-pollock", "hieronymus-bosch",
    "titian", "pieter-bruegel-the-elder", "michelangelo", "marc-chagall", "wassily-kandinsky",
    "kazimir-malevich", "jean-michel-basquiat", "edouard-manet", "camille-pissarro", "georgia-o-keeffe",
    "rene-magritte", "edvard-munch", "thomas-gainsborough", "el-greco", "hans-holbein-the-younger",
    "gustave-courbet", "john-singer-sargent", "albrecht-durer", "georges-braque", "jmw-turner",
    "henri-rousseau", "fernand-leger", "amedeo-modigliani", "egon-schiele", "paul-signac",
    "william-adolphe-bouguereau", "theodore-gericault", "fra-angelico", "canaletto", "antoine-watteau",
    "childe-hassam", "jose-clemente-orozco", "max-beckmann", "bernini", "giotto",
    "hans-memling", "piet-mondrian", "giovanni-bellini", "arnold-bocklin", "frederic-leighton",
    "edwin-lord-weeks", "lucian-freud", "otto-dix", "richard-serra", "maurits-cornelis-escher",
    "jacopo-tintoretto", "henry-moore", "ingres", "odilon-redon", "lorenzo-ghiberti",
    "robert-delaunay", "francisco-de-zurbaran", "constantin-brancusi", "umberto-boccioni", "antoine-blanchard",
    "paolo-veronese", "rogier-van-der-weyden", "giorgione", "gustave-moreau", "carlos-cruz-diez",
    "victor-vasarely", "max-ernst", "alonso-cano", "charles-demuth", "ben-shahn",
    "emil-nolde", "mary-cassatt", "ansel-adams", "joaquin-sorolla", "franz-marc",
    "walter-sickert", "felix-vallotton", "otto-mueller", "hans-hofmann", "frank-stella",
    "claude-lorrain", "theodore-rousseau", "edouard-detaille", "alonso-sanchez-coello", "annibale-carracci",
    "georges-de-la-tour", "peter-paul-rubens", "diego-velazquez", "nicolas-poussin", "andre-derain",
    "alfred-sisley", "federico-barocci", "edouard-vuillard", "constantin-guys", "lawrence-alma-tadema",
    "paul-klee", "milton-avery", "balthus", "morris-louis", "richard-hamilton",
    "jasper-johns", "frank-auerbach", "george-grosz", "victor-brauner", "hans-bellmer",
    "juan-gris", "francis-bacon", "andy-goldsworthy", "yves-klein", "bridget-riley",
    "roy-lichtenstein", "paul-delvaux", "fernando-botero", "jean-dubuffet", "lucio-fontana",
    "yoko-ono", "barbara-kruger", "donald-judd", "sol-lewitt", "walter-gropius",
    "wolfgang-tillmans", "peter-doig", "julian-opie", "kerry-james-marshall", "cindy-sherman",
    "nan-goldin", "ai-weiwei", "takashi-murakami", "damien-hirst", "jeff-koons",
    "gerhard-richter", "anselm-kiefer", "louise-bourgeois", "marina-abramovic", "banksy",
    "shepard-fairey", "ed-ruscha", "richard-prince", "mona-hatoum", "do-ho-suh",
    "sonia-delaunay", "jean-arp", "sophie-taeuber-arp", "arshile-gorky", "barnett-newman",
    "mark-rothko", "willem-de-kooning", "alberto-giacometti", "jean-tinguely", "niki-de-saint-phalle",
    "yayoi-kusama", "zao-wou-ki", "tarsila-do-amaral", "romero-britto", "faith-ringgold",
    "kehinde-wiley", "njideka-akunyili-crosby", "mickalene-thomas", "lynettte-yiadom-boakye", "toyin-ojih-odutola",
    "ai-weiwei", "matthew-barney", "tracey-emin", "olafur-eliasson", "jenny-holzer",
    "michael-craig-martin", "richard-tuttle", "thomas-struth", "thomas-demand", "andreas-gursky",
    "candida-hofer", "gregor-schnieder", "thomas-ruff", "wolfgang-laib", "rebecca-horn",
    "peter-halley", "julian-schnabel", "kiki-smith", "georg-baselitz", "sigmar-polke",
    "martin-kippenberger", "rosemarie-trockel", "karla-black", "thomas-hirschhorn", "sarah-lucas",
    "gary-hume", "cornelia-parker", "rachel-whiteread", "chris-ofili", "damien-hirst",
    "angela-de-la-cruz", "douglas-gordon", "mat-collishaw", "angus-fairhurst", "tracey-emin",
    "francis-als", "gabriel-orozco", "santiago-sierra", "carlos-amorales", "teresa-margolles",
    "mel-chorus", "lothar-baumgarten", "hans-haacke", "dan-graham", "robert-smithson",
    "nancy-holt", "michael-heizer", "vito-acconci", "gordon-matta-clark", "alice-aycock",
    "walid-raad", "shirin-neshat", "mona-hatoum", "bill-viola", "james-turrell",
    "doug-aitken", "ann-hamilton", "tara-donovan", "matthew-ritchie", "fred-tomaselli"
]

csv_filename = 'all_artists_paintings2.csv'

# Crear el archivo CSV vacío si no existe
if not os.path.exists(csv_filename):
    df_empty = pd.DataFrame(columns=["title", "artist_name", "style", "genre", "description"])
    df_empty.to_csv(csv_filename, index=False, encoding='utf-8')

for artist_url in artists:
    print(f"\n--- Descargando obras de {artist_url} ---")
    paintings = get_paintings_with_description(artist_url)

    if paintings:
        df_artist = pd.DataFrame(paintings)
        df_artist.to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8')
        print(f"Se han añadido {len(df_artist)} obras de {artist_url} al CSV.")
    else:
        print(f"No se encontraron obras con descripción para {artist_url}")

print("\n✅ Proceso terminado. Todas las obras han sido añadidas progresivamente a all_artists_paintings.csv")