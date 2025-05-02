import pandas as pd
import matplotlib.pyplot as plt
import os

# Cargar el dataset unificado o crear uno vacío si no existe
if os.path.exists('all_artists_paintings2.csv'):
    df = pd.read_csv('all_artists_paintings2.csv')
else:
    df = pd.DataFrame(columns=['title', 'artist_name', 'style', 'genre', 'description'])

# Mostrar las primeras 5 obras cargadas
if not df.empty:
    echo_df_head = df.head(5)
    print("Primeras 5 obras cargadas:")
    print(echo_df_head[['artist_name', 'title','description']])

    # Mostrar las últimas 5 obras cargadas
    echo_df_tail = df.tail(5)
    print("\nÚltimas 5 obras cargadas:")
    print(echo_df_tail[['artist_name', 'title']])

    # Calcular estadísticas básicas
    num_artists = df['artist_name'].nunique()
    num_styles = df['style'].nunique()
    num_genres = df['genre'].nunique()

    print(f"\nResumen del dataset:")
    print(f"Total de artistas únicos: {num_artists}")
    print(f"Total de estilos únicos: {num_styles}")
    print(f"Total de géneros únicos: {num_genres}")
else:
    print("El dataset está vacío. No hay obras para mostrar.")

# Representar la frecuencia de obras por artista
if not df.empty:
    plt.figure(figsize=(12, 6))
    artist_counts = df['artist_name'].value_counts()
    artist_counts.plot(kind='bar', color='skyblue')
    plt.title('Frecuencia de obras por artista')
    plt.xlabel('Artista')
    plt.ylabel('Cantidad de obras')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Representar la frecuencia de obras por estilo
    plt.figure(figsize=(12, 6))
    style_counts = df['style'].value_counts()
    style_counts.plot(kind='bar', color='lightgreen')
    plt.title('Frecuencia de obras por estilo')
    plt.xlabel('Estilo')
    plt.ylabel('Cantidad de obras')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Representar la frecuencia de obras por género
    plt.figure(figsize=(12, 6))
    genre_counts = df['genre'].value_counts()
    genre_counts.plot(kind='bar', color='salmon')
    plt.title('Frecuencia de obras por género')
    plt.xlabel('Género')
    plt.ylabel('Cantidad de obras')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("El dataset está vacío. No hay datos para representar.")

