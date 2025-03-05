# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# %%
# Me esta dando fallos la libreria dynamo por lo que la voy a desabilitar, si a ti no te da problema puedes omitir estas lineas

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# %%

# Cargar el tokenizador y el modelo si ya lo tenemos gurdado localmente
model_name = "facebook/bart-large-cnn"

device = "cuda"

print("Devise = ",device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

#Definimos un token para el padding
tokenizer.pad_token = tokenizer.eos_token

# Actualizar la configuración del modelo para que use el mismo token de padding
model.config.pad_token_id = tokenizer.pad_token_id

# %%
imput_text = "Climate change is one of the most pressing issues facing humanity today. Scientists have wa"
imput = tokenizer(imput_text , return_tensors="pt").to(device)
output = model.generate(
    **imput,     
    max_length=100,  
    num_return_sequences=1
    )
response = tokenizer.decode(output[0])
print(response)

# %%

from datasets import load_dataset

# Cargargamos el dataset 

dataset = load_dataset('abisee/cnn_dailymail', '3.0.0')

print(dataset)

train_dataset = dataset["train"]
test_dataset = dataset["test"]
eval_dataset = dataset["validation"]

# Seleccionar una porción del dataset de entrenamiento

train_dataset = train_dataset.select(range(int(0.003 * len(train_dataset))))
test_dataset = test_dataset.select(range(int(0.01 * len(test_dataset))))
eval_dataset = eval_dataset.select(range(int(0.01 * len(eval_dataset))))

print(train_dataset)

# %%

def preprocess_function(examples):
    # Combinar el artículo y los highlights con un token especial
    inputs = ["Summarize this article:\n "+ article + "Summary:\n" for article in examples["article"]]

    # Tokenizar los inputs y los targets
    model_inputs = tokenizer(inputs, truncation=True,max_length=512, padding="max_length",return_tensors="pt")
    labels = tokenizer(examples["highlights"], truncation=True,max_length=512, padding="max_length",return_tensors="pt")

    
    # Asignar los labels al diccionario de inputs
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Aplicar la función de preprocesamiento al dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = test_dataset.map(preprocess_function, batched=True)
test_dataset = eval_dataset.map(preprocess_function, batched=True)


data = train_dataset.remove_columns(["id", "highlights","article"])

print(data)

# %%

from transformers import Trainer, TrainingArguments

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),  # Usar mixed precision si hay GPU disponible
)

# Definir el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# Entrenar el modelo
trainer.train()

# %%

# Evaluar el modelo en el conjunto de prueba
results = trainer.evaluate(test_dataset)
print(results)

# %%

# Guardar el modelo y el tokenizador
model.save_pretrained("C:/Users/pablo/ModelosLLM/Bart-FT")
tokenizer.save_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorBart-FT")

# %%

def generate_summary(text,model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=550,
        num_return_sequences=2,
        do_sample=True,
        top_k=5
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

article = "Sumarize this article:\nClimate change is one of the most pressing issues facing humanity today. Scientists have warned that rising global temperatures, caused by increased greenhouse gas emissions, could lead to catastrophic consequences such as extreme weather events, rising sea levels, and the loss of biodiversity. To combat this, countries around the world are adopting measures like transitioning to renewable energy sources, improving energy efficiency, and protecting forests. However, experts emphasize that individual actions, such as reducing waste and using public transportation, are also crucial in the fight against climate change. The time to act is now, as delaying action will only make the problem more difficult and expensive to solve.Summary:\n"
print(generate_summary(article,model))


