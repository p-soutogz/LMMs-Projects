# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# %%
# Me esta dando fallos la libreria dynamo por lo que la voy a desabilitar, si a ti no te da problema puedes omitir esta celda.

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

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
# %%

# Guardar el modelo y el tokenizador
model_base.save_pretrained("C:/Users/pablo/ModelosLLM/Bart")
tokenizer.save_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorBart")

# %%
imput_text = "Sumarize this article:\nClimate change is one of the most pressing issues facing humanity today. Scientists have warned that rising global temperatures, caused by increased greenhouse gas emissions, could lead to catastrophic consequences such as extreme weather events, rising sea levels, and the loss of biodiversity. To combat this, countries around the world are adopting measures like transitioning to renewable energy sources, improving energy efficiency, and protecting forests. However, experts emphasize that individual actions, such as reducing waste and using public transportation, are also crucial in the fight against climate change. The time to act is now, as delaying action will only make the problem more difficult and expensive to solve.Summary:\n"
imput = tokenizer(imput_text , return_tensors="pt").to(device)
output = model_base.generate(
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

train_dataset = train_dataset.select(range(int(0.01 * len(train_dataset))))
test_dataset = test_dataset.select(range(int(0.01 * len(test_dataset))))
eval_dataset = eval_dataset.select(range(int(0.01 * len(eval_dataset))))

print(train_dataset)

# %%

def preprocess_function(examples):
    # Combinar el artículo y los highlights con un token especial
    inputs = ["Summarize this article:\n "+ article + "Summary:\n" for article in examples["article"]]

    # Tokenizar los inputs y los targets
    model_inputs = tokenizer(inputs, truncation=True,max_length=1024, padding="max_length",return_tensors="pt")
    labels = tokenizer(examples["highlights"], truncation=True,max_length=250, padding="max_length",return_tensors="pt")

    
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
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Usar mixed precision si hay GPU disponible
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


# %%
# Entrenar el modelo
trainer_base.train()


# %%

# Guardar el modelo 
##model_base.save_pretrained("C:/Users/pablo/ModelosLLM/Bart-FT")
# %%

#Cargar el modelo
 
model_FullFineTuning = AutoModelForSeq2SeqLM.from_pretrained("C:/Users/pablo/ModelosLLM/Bart-FT").to(device)
tokenizer = AutoTokenizer.from_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorBart-FT")

# %%

def generate_summary(text,model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=250,
        min_new_tokens=50,
        num_return_sequences=1,
        do_sample=True,
        top_k=5
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
# %%
article = "Sumarize this article:\nClimate change is one of the most pressing issues facing humanity today. Scientists have warned that rising global temperatures, caused by increased greenhouse gas emissions, could lead to catastrophic consequences such as extreme weather events, rising sea levels, and the loss of biodiversity. To combat this, countries around the world are adopting measures like transitioning to renewable energy sources, improving energy efficiency, and protecting forests. However, experts emphasize that individual actions, such as reducing waste and using public transportation, are also crucial in the fight against climate change. The time to act is now, as delaying action will only make the problem more difficult and expensive to solve.Summary:\n"

print(generate_summary(article,model_base))
print(generate_summary(article,model_FullFineTuning))

# %%
#Vamos ahora a comparar la eficacia de los dos modelos sobre el dataset de evaluacion usando Trainer

trainer_ft = Trainer(
    model=model_FullFineTuning,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
# %%

results_base_model = trainer_base.evaluate(test_dataset)
results_ft_model = trainer_ft.evaluate(test_dataset)

print("Resultados modelo base:\n")
print(results_base_model)
print("Resultdos modelo finetuning:\n")
print(results_ft_model)



# %%
## Vamos a compararlos usando la metrica ROUGE

from evaluate import load

eval_data = eval_dataset.select(range(50))

rouge = load("rouge",token=True)

references = [example["highlights"] for example in eval_data]  

# %%
def compute_predictions(data, model, tokenizer, device):
    predictions = []
    
    for i in range(0, len(data)):
        articles = "Sumarize this article:\n"+data["article"][i]+ "Summary:\n"

        inputs = tokenizer(
            articles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        summary_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            num_return_sequences=1,
            do_sample=True,
            top_k=5
        )
        
        summaries = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(summaries)
    
    return predictions


# %%
predictions_base = compute_predictions(eval_data,model_base, tokenizer, device)
rouge_base = rouge.compute(predictions=predictions_base, references=references)
print("ROUGE del modelo base:", rouge_base)


# %%

predictions_tf = compute_predictions(eval_data,model_FullFineTuning, tokenizer, device)
rouge_ft = rouge.compute(predictions=predictions_tf, references=references)
print("ROUGE del modelo fine-tuneado:", rouge_ft)






























