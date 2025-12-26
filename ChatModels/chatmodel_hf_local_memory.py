from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    task="text-generation",
    model=model_id,
    tokenizer=model_id,
    max_new_tokens=50,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

# ✅ Memory
memory = ConversationBufferMemory()

# ✅ Conversation chain
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=True
)

# ---- Chat loop ----
while True:
    query = input("Ask: ")
    response = conversation.predict(input=query)
    print(response)

    if input("Continue? (y/n): ").lower() == "n":
        break
