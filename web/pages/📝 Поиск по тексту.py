import streamlit as st
import base64
from PIL import Image
from io import BytesIO
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import HuggingFaceHub


def get_image(data):
    image = Image.open(BytesIO(base64.b64decode(data))).convert('RGB')
    return image


model = 'tomaarsen/mpnet-base-nli-matryoshka'

embeddings = HuggingFaceEmbeddings(model_name=model)
db = FAISS.load_local("./Inference_models/faiss", embeddings, allow_dangerous_deserialization=True)
YOUR_API_KEY = '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'
# создаем шаблон для промта
prompt_template = """Используй контекст для ответа на вопрос, пользуясь следующими правилами:

Не изменяй текст, который находится в кавычках.
В конце обязательно добавь ссылку на полный документ
{answer}
img: {image}
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=['answer', 'image'])
chain = LLMChain(prompt=PROMPT, llm=HuggingFaceHub(
                                    repo_id='IlyaGusev/fred_t5_ru_turbo_alpaca',
                                    huggingfacehub_api_token=YOUR_API_KEY,
                                    model_kwargs={'temperature':0, 'max_length':128})
)

st.set_page_config(
    page_title='Поиск по тексту',
    page_icon="📝"
)
st.title('Нахождение места по описанию')

txt = st.text_area(
    "Ваше описание",
    "",
    )
if len(txt):
    relevants = db.similarity_search(txt)
    doc = relevants[0].dict()['metadata']
    t = chain.run(doc)
    st.write(t.split('\n')[-3])
    st.image(get_image(t.split('\n')[-2][5:]))
