import os
from dotenv import load_dotenv
from llama_index import GPTVectorStoreIndex, download_loader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")

# load documents
TwitterTweetReader = download_loader("TwitterTweetReader")
loader = TwitterTweetReader(bearer_token=TWITTER_API_BEARER_TOKEN)
# documents = loader.load_data(twitterhandles=['elonmusk'], num_tweets=10)
documents = loader.load_langchain_documents(
    twitterhandles=['elonmusk'], num_tweets=10)

# documentを見る
# print(documents)

# CREATE SAMPLE CHAIN for longterm memory
# initialize sample chain
llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm)
query = "How fast did the fairings in the ViaSat-3 mission fall, relative to the speed of sound?"
answer = qa_chain.run(input_documents=documents, question=query)
print(answer)
