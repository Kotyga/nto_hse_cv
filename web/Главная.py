import streamlit as st

st.set_page_config(
    page_title='Главная',
    page_icon="🏠"
)
st.balloons()
st.page_link("Главная.py", label="Home", icon="🏠")
st.page_link("./pages/📝 Поиск по тексту.py", label="Поиск по тексту", icon="📝")
st.page_link("./pages/🖼️ Поиск по фото.py", label="Поиск по фото", icon="🖼️")
st.page_link("./pages/🗺️ Маршрут.py", label="Маршрут", icon="🗺️")
st.page_link("./pages/🤝 О нас.py", label="О нас", icon="🤝")

st.title('ExploreMe: Ваш гид в мире приключений!')
st.write('''Открой для себя мир новых приключений с сервисом ExploreMe!
         Просто сфотографируй интересное место, чтобы узнать больше о нем, или введите описание,
         чтобы найти наименование и снимок его достопримечательности.
         А если хочешь исследовать новые места,
         просто укажи координаты - и мы построим для тебя самый удивительный маршрут.
         Погнали в путешествие вместе с ExploreMe!''')