import streamlit as st

# Настройка навигации
pages = {
    "Анализ и модель": "analysis_and_model.py",
    "Презентация": "presentation.py",
}

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти к", list(pages.keys()))

page = pages[selection]

# Динамический импорт выбранной страницы
if page == "analysis_and_model.py":
    from analysis_and_model import analysis_and_model_page
    analysis_and_model_page()
elif page == "presentation.py":
    from presentation import presentation_page
    presentation_page()