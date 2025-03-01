# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# He organizado el codigo en celdas para mayor comodidad
#Este modelo esta entrenado especialmente para traducir

# %%
model_name = "ModelSpace/GemmaX2-28-2B-v0.1"

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#Definimos un token para el padding
tokenizer.pad_token = tokenizer.eos_token

# Actualizar la configuración del modelo para que use el mismo token de padding
model.config.pad_token_id = tokenizer.pad_token_id
# %%
#Guardamos el modelo y el tokenizador localmente
ruta_modelo = "C:/Users/pablo/ModelosLLM/GemmaX2"
ruta_tokenizador = "C:/Users/pablo/ModelosLLM/GemmaX2-tokenizador"

model.save_pretrained(ruta_modelo)
tokenizer.save_pretrained(ruta_tokenizador)

# %%

input_text="Translate this from Chinese to English:\nChinese: 我爱机器翻译\nEnglish:"

inputs = tokenizer(input_text,return_tensors="pt")

output=model.generate(
    **inputs,
    max_length=100,  
    num_return_sequences=1,  
    )
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Input:: {input_text}")
print(f"Generated Text: {generated_text}")
