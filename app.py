from fastapi import Request,FastAPI
from pydantic import BaseModel
import math 
import ThreadPoolExecutor
import uvicorn

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


app = FastAPI()

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)


tokenizer = AutoTokenizer.from_pretrained('./bart-large-cnn-samsum')
model = AutoModelForSeq2SeqLM.from_pretrained('./bart-large-cnn-samsum').to(torch_device)
model = model.to(torch_device)


class SummaryRequest(BaseModel):
    text: str

def summarise_text(text, tokeniser_summariser , model_summariser):

    # Define the input text
    input_text = text["text"]

    summarizer = pipeline("summarization", model=model_summariser , tokenizer= tokeniser_summariser)
    # Define the maximum token limit per chunk
    max_tokens_per_chunk = 516

    # Tokenize the input text
    tokens = tokeniser_summariser.tokenize(input_text)

    # Calculate the total number of chunks
    num_chunks = math.ceil(len(tokens) / max_tokens_per_chunk)

    # Define the summary length
    max_summary_length = 100
    min_summary_length = 20

    # Define the number of threads to use
    num_threads = 4

    # Define a function to generate a summary for a given chunk of text
    def summarize_chunk(chunk_tokens):
        chunk_text = tokeniser_summariser.convert_tokens_to_string(chunk_tokens)
        return summarizer(chunk_text, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)[0]['summary_text']

    # Split the input text into chunks and generate summaries in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        summaries = list(executor.map(summarize_chunk, [tokens[start:end] for start, end in zip(range(0, len(tokens), max_tokens_per_chunk), range(max_tokens_per_chunk, len(tokens) + max_tokens_per_chunk, max_tokens_per_chunk))]))
        
    # Join the summaries together into a single string
    final_summary = " ".join(summaries)

    # Generate a final summary for the entire input text
    result = summarizer(final_summary, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)[0]['summary_text']
    return {"summary": result}

 

@app.get('/')
async def home():
    return {"message": "Hello World , It's Vanine live from AWS !"}

@app.post("/summary")
async def getsummary(user_request_in: SummaryRequest):
    payload = {"text":user_request_in.text}
    summ = summarise_text(payload,tokenizer,model)
    summ["Device"]= torch_device
    return 
