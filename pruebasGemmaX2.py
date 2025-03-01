# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# He organizado el codigo en celdas para mayor comodidad

# %%
model_name = "ModelSpace/GemmaX2-28-2B-v0.1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)

#Definimos un token para el padding
tokenizer.pad_token = tokenizer.eos_token

# Actualizar la configuración del modelo para que use el mismo token de padding
model.config.pad_token_id = tokenizer.pad_token_id

#Guardamos el modelo y el tokenizador localmente
ruta_modelo = "C:/Users/pablo/ModelosLLM/GemmaX2"
ruta_tokenizador = "C:/Users/pablo/ModelosLLM/GemmaX2-tokenizador"

model.save_pretrained(ruta_modelo)
tokenizer.save_pretrained(ruta_tokenizador)

# %%

input_text="Hola que tal estas"

inputs = tokenizer(input_text,return_tensors="pt").to(device)

output=model.generate(
    **inputs,
    max_length=100,  
    num_return_sequences=1,  
    )
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
