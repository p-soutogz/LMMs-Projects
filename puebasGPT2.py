# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer

# He organizado el codigo en celdas para mayor comodidad

# %%
model_name = "openai-community/gpt2"

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#Guardamos el modelo y el tokenizador localmente
model.save_pretrained("C:/Users/pablo/GPT2")
tokenizer.save_pretrained("C:/Users/pablo/tokenizadorGPT2")

#Definimos un token para el padding
tokenizer.pad_token = tokenizer.eos_token

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

# Vemos que este modelo no responde de la forma mas deseada además tiende a ser repetitivo. Esto ultimo podemos 
# mejorarlo introduciendo un poco de aleatoriedad en la generacion de tokens

# %%

# Generar texto

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],  
    max_length=50,  
    do_sample=True,  # Habilitar sampling
    top_k=50,  # Limitar la selección a los 50 tokens más probables
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

# Ya no es tan repetitivo pero sigue sin ser satisfactorio, además que ahora se ve claramente que el modelo  
# ha sido entrenado en un corpus exclusivamente en ingles luego no responde de la forma deseada a los inputs en castellano
