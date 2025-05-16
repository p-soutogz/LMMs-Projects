import openai
import pandas as pd
import time
import os

# Tu clave de API 


# Nombre del archivo de trabajo
ruta_backup = "backup.csv"
ruta_original = "prompts_chatgpt_artworks_filtered_minlen300.csv"

# Cargar dataset: backup si existe, si no el original
if os.path.exists(ruta_backup):
    print("🔁 Cargando backup existente...")
    df = pd.read_csv(ruta_backup)
else:
    print("🆕 No se encontró backup. Cargando dataset original y creando uno nuevo...")
    df = pd.read_csv(ruta_original)
    df['output'] = ''
    df.to_csv(ruta_backup, index=False)

# Confirmar número total de filas
total_filas = len(df)
print(f"📂 Total de obras a procesar: {total_filas}\n")

# Procesar fila por fila solo si no tiene resultado previo
for i, row in df.iterrows():
    salida = str(df.at[i, 'output'])

    if salida.strip() == '' or salida.lower() == 'nan' or salida.startswith("ERROR"):
        prompt = row['input']

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512
            )
            answer = response['choices'][0]['message']['content']
            df.at[i, 'output'] = answer
            print(f"✅ {i+1}/{total_filas} completado correctamente.")
        except Exception as e:
            df.at[i, 'output'] = f"ERROR: {str(e)}"
            print(f"❌ Error en fila {i+1}: {e}")

        # Guardar tras cada petición
        df.to_csv(ruta_backup, index=False)
        time.sleep(21)  # Delay para respetar el límite

print("\n🏁 Proceso terminado. El archivo 'backup.csv' contiene los resultados.")


