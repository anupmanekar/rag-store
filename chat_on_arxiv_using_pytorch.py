# This program demonstrates RAG functionality using PyTorch and Hugging Face Transformers rather than LangChain or LlamaIndex like frameworks
# The program uses the subset of arXiv papers dataset from MongoDB and performs a search query to get the relevant papers
# The program uses the Sentence Transformers library to generate embeddings for the dataset and the user query
# The program uses the Meta-LLama-3-8B-Instruct model to generate responses to the user query


# Load Dataset
import os
import torch
import pandas as pd
from datasets import load_dataset
from rag_utils.utils import get_embedding, get_mongo_client, get_search_result
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

load_dotenv()
login(token = '')

dataset = load_dataset("MongoDB/subset_arxiv_papers_with_embeddings")
dataset_df = pd.DataFrame(dataset["train"])
print(dataset_df.head())

# Prepare dataset to consider only first 100 samples and remove the embeddings column
dataset_df = dataset_df.head(100)
dataset_df = dataset_df.drop(columns=["embedding"])
print(dataset_df.head())

# Generate embeddings from 3 columns title, authors, abstract and insert into the dataset as a new column
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('thenlper/gte-large')

dataset_df["embedding"] = dataset_df.apply(
    lambda x: get_embedding(x["title"] + " " + x["authors"] + " " + x["abstract"], embedding_model=model),
    axis=1,
)

print(dataset_df.head())

mongo_uri = os.getenv("MONGO_URI")
client = get_mongo_client(mongo_uri)
db = client.get_database("knowledge_base")
collection = db.get_collection("research_papers")

# Delete the existing data from the collection
collection.delete_many({})

# Insert the dataset into the MongoDB collection
collection.insert_many(dataset_df.to_dict(orient="records"))
print("Data inserted successfully")

# Perform a search query
query = "Get me papers on Artificial Intelligence?"
source_information = get_search_result(query, collection, embedding_model=model)
combined_information = f"Query: {query}\nContinue to answer the query by using the Search Results:\n{source_information}."
messages = [
    {"role": "system", "content": "You are a research assitant!"},
    {"role": "user", "content": combined_information},
]
print(messages)

# Search using LLama

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))

# Close the MongoDB connection
client.close()
print("Connection to MongoDB closed")