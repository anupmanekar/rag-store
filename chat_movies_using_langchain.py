import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_fireworks import ChatFireworks, FireworksEmbeddings

from rag_utils.utils import get_mongo_client

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "movies_db"
COLLECTION_NAME = "data"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

data = load_dataset("MongoDB/embedded_movies")
df = pd.DataFrame(data["train"])
print (df.head())

# Only keep records where the fullplot field is not null
df = df[df['fullplot'].notnull()]

df.rename(columns={"plot_embedding": "embedding"}, inplace=True)

client = get_mongo_client(MONGO_URI)
db = client.get_database(DB_NAME)
collection = db.get_collection(COLLECTION_NAME)

# Delete any existing records in the collection
collection.delete_many({})

# Data Ingestion
records = df.to_dict("records")
collection.insert_many(records)

print("Data ingestion into MongoDB completed")

embeddings = FireworksEmbeddings(model='nomic-ai/nomic-embed-text-v1.5')

# Vector Store Creation
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="fullplot",
)

print("Vector Store created successfully")

# Using the MongoDB vector store as a retriever in a RAG chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Generate context using the retriever, and pass the user question through
retrieve = {
    "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
    "question": RunnablePassthrough(),
}
template = """Answer the question based only on the following context: \
{context}

Question: {question}
"""

# Defining the chat prompt
prompt = ChatPromptTemplate.from_template(template)
# Defining the model to be used for chat completion
model = ChatFireworks(model="accounts/fireworks/models/firefunction-v1", max_tokens=256, api_key=FIREWORKS_API_KEY)
# Parse output as a string
parse_output = StrOutputParser()

# Naive RAG chain
naive_rag_chain = retrieve | prompt | model | parse_output

naive_rag_chain.invoke("What is the best movie to watch when sad?")

