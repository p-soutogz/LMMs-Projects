# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    "I just returned from the greatest summer vacation! It was so fantastic, I never wanted it to end. I spent eight days in Paris, France. My best friends, Henry and Steve, went with me. We had a beautiful hotel room in the Latin Quarter, and it wasn’t even expensive. We had a balcony with a wonderful view.",
    "Cual es la capital de España"
]
inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side='left',truncation=True, return_attention_mask=True)
print(inputs)
# %%

# Generar texto

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],  
    max_length=75,  
    num_return_sequences=1,  
)
print(outputs)
# %%
# Decodificar y mostrar el texto generado
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(generated_texts)
# %%