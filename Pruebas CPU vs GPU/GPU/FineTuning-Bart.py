# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

# %%

# Cargar el tokenizador y el modelo si ya lo tenemos gurdado localmente
model_name = "facebook/bart-large-cnn"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Devise = ",device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model_base = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

#Definimos un token para el padding
tokenizer.pad_token = '<PAD>'

# Actualizar la configuración del modelo para que use el mismo token de padding
model_base.config.pad_token_id = tokenizer.pad_token_id


#%% Cargar el dataset

from datasets import load_dataset,DatasetDict

dataset = load_dataset('gopalkalpande/bbc-news-summary')

split_dataset = dataset["train"].train_test_split(
    test_size=0.2,
    shuffle=True,
    seed=42
)

train_dataset = split_dataset["train"].select(range(int(0.1 * len(split_dataset["train"]))))
eval_dataset = split_dataset["test"].select(range(int(0.1 * len(split_dataset["test"]))))


# %%

def preprocess_function(examples):
    # Combinar el artículo y los highlights con un token especial
    inputs = ["Summarize this article:\n "+ article + "Summary:\n" for article in examples["Articles"]]

    # Tokenizar los inputs y los targets
    model_inputs = tokenizer(inputs, truncation=True,max_length=1024, padding="max_length",return_tensors="pt")
    labels = tokenizer(examples["Summaries"], truncation=True,max_length=250, padding="max_length",return_tensors="pt")

    
    # Asignar los labels al diccionario de inputs
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Aplicar la función de preprocesamiento al dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

data = train_dataset.remove_columns(["File_path", "Summaries","Articles"])

print(data)

# %%

from transformers import Trainer, TrainingArguments

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    )

# %%
# Definir el Trainer
trainer_base = Trainer(
    model=model_base,
    args=training_args,
    train_dataset=data,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)


#%% Entrenar el modelo
start_time = time.time() 
trainer_base.train()
end_time = time.time()  

training_duration = end_time - start_time

with open("resultados_entrenamiento_BartFFT.txt", "w") as archivo:
    archivo.write(f"\nTiempo total de entrenamiento: {training_duration:.2f} segundos\n")
    archivo.write(f"Tiempo por época: {training_duration/training_args.num_train_epochs:.2f} segundos\n")

# %%

# Guardar el modelo y el tokenizador(Modifica tu la ruta)
model_base.save_pretrained("C:/Users/pablo/ModelosLLM/Bart-FT")
tokenizer.save_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorBart-FT")
 



























