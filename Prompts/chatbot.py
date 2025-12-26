from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

model_id = "HuggingFaceTB/SmolLM-135M"

pipe = pipeline(
    task="text-generation",
    model=model_id,
    tokenizer=model_id,
    max_new_tokens=50,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)