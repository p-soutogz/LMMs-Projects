from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification,BertTokenizer,BertModel

classifier= pipeline("sentiment-analysis")

res = classifier("I really love hugging face")

model_name="distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("I really love hugging face")

print(res)