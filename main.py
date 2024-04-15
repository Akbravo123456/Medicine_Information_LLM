import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

st.title('Medicine Information Search')

input_text = st.text_input("Search for a medicine:")

first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about medicine {name}"
)

medicine_memory = ConversationBufferMemory(input_key='name', memory_key='medicine_chat_history')
description_memory = ConversationBufferMemory(input_key='medicine', memory_key='medicine_description_history')

llm = OpenAI(temperature=0.8)

chain1 = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    verbose=True,
    output_key='medicine',
    memory=medicine_memory
)

second_input_prompt = PromptTemplate(
    input_variables=['medicine'],
     template="Mention 5 major events related to {medicine}"
)

chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    verbose=True,
    output_key='description',
    memory=description_memory
)

parent_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=['name'],
    output_variables=['medicine', 'description'],
    verbose=True
)

if input_text:
    st.write(parent_chain({'name': input_text}))

    with st.expander('Medicine Information'): 
        st.info(medicine_memory.buffer)

    with st.expander('Major Events'): 
        st.info(description_memory.buffer)