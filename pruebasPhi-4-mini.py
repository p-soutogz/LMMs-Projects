# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer

# He organizado el codigo en celdas para mayor comodidad

#Este modelo es considerablemente mas grande que GPT2 y ademas ha sido fine-tunneado 
#en un data base orientado a la generacion de texto

#No recomiendo ejecutar el codigo, es bastante exigente

# %%
model_name = "unsloth/Phi-4-mini-instruct"

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#Definimos un token para el padding
tokenizer.pad_token = tokenizer.eos_token

# Actualizar la configuración del modelo para que use el mismo token de padding
model.config.pad_token_id = tokenizer.pad_token_id

#Guardamos el modelo y el tokenizador localmente
model.save_pretrained("C:/Users/pablo/Phi-4-mini")
tokenizer.save_pretrained("C:/Users/pablo/Phi-4-mini-tokenizador")

# %%
# Texto de entrada
input_texts = [
    "Tell me the capital of Spain",
    "I just returned from the greatest summer vacation!",
    "Cual es la capital de España",
    "2+3="
]
inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side='left',truncation=True, return_attention_mask=True)

# %%

# Generar texto

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],  
    max_length=50,  
    num_return_sequences=1,  
)

# %%
# Decodificar y mostrar el texto generado
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# Mostrar los textos generados
for i, text in enumerate(generated_texts):
    print(f"Input {i+1}: {input_texts[i]}")
    print(f"Generated Text {i+1}: {text}")
    print("-" * 50)

# %%
#La salida generada fue la siguiente:

#Input 1: Tell me the capital of Spain
#Generated Text 1: Tell me the capital of Spain. The capital of Spain is Madrid.
#--------------------------------------------------
#Input 2: I just returned from the greatest summer vacation!
#Generated Text 2: I just returned from the greatest summer vacation! I spent my days exploring the beautiful beaches, hiking through lush forests, and enjoying the local cuisine. The weather was perfect, and I made some unforgettable memories. I can't wait to share all the details with
#--------------------------------------------------
#Input 3: Cual es la capital de España
#Generated Text 3: Cual es la capital de España? La capital de España es Madrid. Madrid es la ciudad más grande del país y sirve como el centro político, económico y cultural de España. Es el lugar donde se encuentra el gobierno español, incluyendo el
#--------------------------------------------------
#Input 4: 2+3=
#Generated Text 4: 2+3=5, 2+4=6, 2+5=7, 2+6=8, 2+7=9, 2+8=10, 2+9
#--------------------------------------------------

#Claramente el modelo responde mucho mejor que GPT2, además este si que esta entrenado tambien en español

# %%
##Vamos ahora a cargar el modelo en la GPU para aguilizar los calculos

# Definir el dispositivo (GPU)
device = "cuda" 

ruta_modelo = "C:/Users/pablo/Phi-4-mini"
ruta_tokenizador = "C:/Users/pablo/Phi-4-mini-tokenizador"

# Cargar el tokenizador y el modelo en la GPU
tokenizer = AutoTokenizer.from_pretrained(ruta_tokenizador)
model = AutoModelForCausalLM.from_pretrained(ruta_modelo).to(device)
# %%

input_texts = [
    "Tell me the capital of Spain",
    "I just returned from the greatest summer vacation!",
    "Cual es la capital de España",
    "2+3="
]
inputs = tokenizer(input_texts, return_tensors="pt",padding=True, padding_side='left',truncation=True, return_attention_mask=True).to(device)

# Generar texto
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],  
    max_length=50,
    num_return_sequences=1  
    )
# %%
# Decodificar y mostrar el texto generado
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# Mostrar los textos generados
for i, text in enumerate(generated_texts):
    print(f"Input {i+1}: {input_texts[i]}")
    print(f"Generated Text {i+1}: {text}")
    print("-" * 50)

## De este modo reducimos notoriamente el tiempo de calculo
