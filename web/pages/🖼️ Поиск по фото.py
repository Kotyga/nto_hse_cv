import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import joblib
from torchvision import models
import plotly.express as px
from transformers import pipeline
import pandas as pd
st.set_page_config(
    page_title='Поиск по фото',
    page_icon="🖼️"
)
uploaded_file = st.file_uploader("Загрузите изображение...", type="png")
d = pd.read_csv('./static/geo.csv')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption='Ваша картинка', use_column_width=True)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    input_tensor = test_transform(image).unsqueeze(0)
    device = torch.device("cpu")
    input_tensor = input_tensor.to(device)

    model_weights_path = './Inference_models/models/resnet_101.pth'
    label_encoder_path = './Inference_models/models/label_encoder.pkl'

    model = models.resnet101()
    model.fc = torch.nn.Linear(model.fc.in_features, 387)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model = model.to(device)

    label_encoder = joblib.load(label_encoder_path)

    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.topk(outputs, 5)
    predicted_indexes = predicted.cpu().numpy()[0]
    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0][predicted_indexes]

    predicted_labels = label_encoder.inverse_transform(predicted_indexes)
    for i in range(len(predicted_labels)):
        st.write(f"Топ-{i+1} предсказание: {predicted_labels[i]}, Вероятность: {round(probabilities[i]*100, 2)}")

    fig = px.bar(x=probabilities, y=predicted_labels, orientation='h', color=probabilities, color_continuous_scale='Blues')
    fig.update_layout(title='Топ-5 Предсказаний', xaxis_title='Вероятность', yaxis_title='Классы')
    st.plotly_chart(fig)

    n = d[d['Name'].isin(predicted_labels)][['Lat', 'Lon']]
    st.map(n, latitude='Lat', longitude='Lon')

    kind_texts = list(d['Kind'])

    result = []

    for string in kind_texts:
        elements = string.split(',')
        result.extend(elements)

    texts = list(set(result))

    image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-256-i18n")
    outputs = image_classifier(image, candidate_labels=texts)
    outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]

    outputs.sort(key=lambda x: x['score'], reverse=True)
    top_5_outputs = outputs[:5]
    # Получаем список вероятностей и лейблов из top_5_outputs
    probabilities = [output["score"] for output in top_5_outputs]
    labels = [output["label"] for output in top_5_outputs]

    fig = px.bar(x=labels,
            y=probabilities,
            color=probabilities,
            color_continuous_scale='Blues')
    fig.update_layout(title='Вероятности для топ-5 лейблов', xaxis_title='Лейблы', yaxis_title='Вероятности')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
