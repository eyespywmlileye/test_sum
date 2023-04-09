
from transformers import AutoTokenizer, pipeline , AutoModelForSeq2SeqLM

model_name = "philschmid/bart-large-cnn-samsum"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.save_pretrained('./bart-large-cnn-samsum')
tokenizer.save_pretrained('./bart-large-cnn-samsum')
