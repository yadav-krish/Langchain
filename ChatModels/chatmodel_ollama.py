from langchain_community.chat_models import ChatOllama

#we have to pull the model if not pulled already "ollama pull model_name"
llm=ChatOllama(model="tinyllama")

response=llm.invoke("Tell me about Putin's recent visit to India")
print(response)