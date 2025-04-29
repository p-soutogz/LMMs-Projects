# -*- coding: utf-8 -*-


# He organizado el codigo en celdas para mayor comodidad

# Este modelo esta entrenado especialmente para traducir

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ruta_modelo = "C:/Users/pablo/ModelosLLM/GemmaX2"
ruta_tokenizador = "C:/Users/pablo/ModelosLLM/GemmaX2-tokenizador"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

# %%

# Me esta dando fallos la libreria dynamo por lo que la voy a desabilitar, si a ti no te da problema puedes omitir estas lineas

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True


# %%
model_name = "ModelSpace/GemmaX2-28-2B-v0.1"

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

#Definimos un token para el padding
tokenizer.pad_token = tokenizer.eos_token

# Actualizar la configuración del modelo para que use el mismo token de padding
model.config.pad_token_id = tokenizer.pad_token_id

#Guardamos el modelo y el tokenizador localmente

model.save_pretrained(ruta_modelo)
tokenizer.save_pretrained(ruta_tokenizador)

# %%

# Cargar el tokenizador y el modelo si ya lo tenemos gurdado localmente

tokenizer = AutoTokenizer.from_pretrained(ruta_tokenizador)
model = AutoModelForCausalLM.from_pretrained(ruta_modelo).to(device)


# %%

input_text="Translate this from Spanish to English:\nSpanish: En un lugar de La Mancha de cuyo nombre no puedo acordarme...\nEnglish:"

inputs = tokenizer(input_text,return_tensors="pt").to(device)

output=model.generate(
    **inputs,
    max_length=100,  
    num_return_sequences=1,  
    )
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Input:: {input_text}")
print(f"Generated Text: {generated_text}")
