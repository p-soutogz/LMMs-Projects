import pandas as pd
import matplotlib.pyplot as plt
import os

# Cargar el dataset unificado o crear uno vacío si no existe
if os.path.exists('all_artists_paintings2.csv'):
    df = pd.read_csv('all_artists_paintings2.csv')
else:
    df = pd.DataFrame(columns=['title', 'artist_name', 'style', 'genre', 'description'])

# Limpieza de datos: valores nulos o vacíos
df['artist_name'] = df['artist_name'].fillna('').astype(str).str.strip().replace('', 'Desconocido')
df['style'] = df['style'].fillna('').astype(str).str.strip().replace('', 'Desconocido')
df['genre'] = df['genre'].fillna('').astype(str).str.strip().replace('', 'Desconocido')
df['description'] = df['description'].fillna('').astype(str).str.strip()

# Mostrar las primeras y últimas 5 obras cargadas
if not df.empty:
    echo_df_head = df.head(5)
    print("Primeras 5 obras cargadas:")
    print(echo_df_head[['artist_name', 'title','description']])

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

    # === Estadísticas adicionales ===
    obras_por_artista = df['artist_name'].value_counts()
    avg_obras = obras_por_artista.mean()
    max_obras = obras_por_artista.max()
    min_obras = obras_por_artista.min()

    print(f"\nEstadísticas de obras por artista:")
    print(f"- Promedio de obras por artista: {avg_obras:.2f}")
    print(f"- Máximo de obras por un artista: {max_obras}")
    print(f"- Mínimo de obras por artista: {min_obras}")

    # Longitud de descripciones
    desc_lengths = df['description'].apply(len)
    avg_desc_length = desc_lengths.mean()
    median_desc_length = desc_lengths.median()

    print(f"\nEstadísticas de longitud de descripciones:")
    print(f"- Longitud media de las descripciones: {avg_desc_length:.2f} caracteres")
    print(f"- Longitud mediana de las descripciones: {median_desc_length:.0f} caracteres")

else:
    print("El dataset está vacío. No hay obras para mostrar.")

# Representaciones gráficas solo si hay datos
if not df.empty:
    # Top 10 artistas
    plt.figure(figsize=(12, 6))
    top_artists = df['artist_name'].value_counts().head(10)
    top_artists.plot(kind='bar', color='skyblue')
    plt.title('Top 10 artistas con más obras')
    plt.xlabel('Artista')
    plt.ylabel('Cantidad de obras')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Top 10 estilos
    plt.figure(figsize=(12, 6))
    top_styles = df['style'].value_counts().head(10)
    top_styles.plot(kind='bar', color='lightgreen')
    plt.title('Top 10 estilos más frecuentes')
    plt.xlabel('Estilo')
    plt.ylabel('Cantidad de obras')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Top 10 géneros
    plt.figure(figsize=(12, 6))
    top_genres = df['genre'].value_counts().head(10)
    top_genres.plot(kind='bar', color='salmon')
    plt.title('Top 10 géneros más frecuentes')
    plt.xlabel('Género')
    plt.ylabel('Cantidad de obras')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("El dataset está vacío. No hay datos para representar.")



