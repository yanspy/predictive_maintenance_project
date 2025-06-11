import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
from ucimlrepo import fetch_ucirepo


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Инициализация состояния модели
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
    if 'scaler_fitted' not in st.session_state:
        st.session_state['scaler_fitted'] = False

    # Загрузка данных
    st.header("1. Загрузка данных")
    data_source = st.radio("Источник данных",
                         ["Пример данных (встроенный)", "Загрузить CSV-файл"])

    data = None
    if data_source == "Пример данных (встроенный)":
        try:
            ai4i_2020 = fetch_ucirepo(id=601)
            data = pd.concat([ai4i_2020.data.features, ai4i_2020.data.targets], axis=1)
            st.success("Встроенный датасет успешно загружен!")
        except Exception as e:
            st.error(f"Ошибка загрузки встроенного датасета: {e}")
    else:
        uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("Файл успешно загружен!")
            except Exception as e:
                st.error(f"Ошибка загрузки файла: {e}")

    if data is not None:
        # Переименование столбцов — удаление скобок и замена на понятные названия
        data.rename(columns={
            'Air temperature [K]': 'Air temperature K',
            'Process temperature [K]': 'Process temperature K',
            'Rotational speed [rpm]': 'Rotational speed rpm',
            'Torque [Nm]': 'Torque Nm',
            'Tool wear [min]': 'Tool wear min'
        }, inplace=True)

        st.write("Первые 5 строк данных:", data.head())

        # Предобработка данных
        st.header("2. Предобработка данных")

        # Удаление ненужных столбцов
        cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        cols_to_drop = [col for col in cols_to_drop if col in data.columns]
        data = data.drop(columns=cols_to_drop)

        # Преобразование категориального признака Type
        if 'Type' in data.columns:
            le = LabelEncoder()
            data['Type'] = le.fit_transform(data['Type'].astype(str))

        # Проверка пропущенных значений
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            data = data.fillna(data.mean())

        # Масштабирование числовых признаков
        numerical_features = ['Air temperature K', 'Process temperature K',
                              'Rotational speed rpm', 'Torque Nm', 'Tool wear min']
        numerical_features = [col for col in numerical_features if col in data.columns]

        if numerical_features:
            scaler = StandardScaler()
            data[numerical_features] = scaler.fit_transform(data[numerical_features])
            st.session_state['scaler_fitted'] = True
            st.session_state['scaler'] = scaler
        else:
            st.error("Не найдены числовые признаки для масштабирования!")
            st.stop()

        # Разделение данных
        st.header("3. Разделение данных")
        test_size = st.slider("Размер тестовой выборки (%)", 10, 40, 20)
        random_state = st.number_input("Random state", value=42)

        try:
            X = data.drop(columns=['Machine failure'])
            y = data['Machine failure']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=random_state)
        except KeyError:
            st.error("Целевая переменная 'Machine failure' отсутствует в данных.")
            return

        # Обучение модели
        st.header("4. Обучение модели")
        model_choice = st.selectbox("Выберите модель",
                                  ["Logistic Regression", "Random Forest",
                                   "XGBoost", "SVM"])

        if model_choice == "Logistic Regression":
            model = LogisticRegression(random_state=random_state)
        elif model_choice == "Random Forest":
            n_estimators = st.slider("Количество деревьев", 50, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators,
                                         random_state=random_state)
        elif model_choice == "XGBoost":
            n_estimators = st.slider("Количество деревьев", 50, 200, 100)
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
            model = XGBClassifier(n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                random_state=random_state)
        elif model_choice == "SVM":
            kernel = st.selectbox("Ядро", ["linear", "rbf", "poly"])
            model = SVC(kernel=kernel, probability=True, random_state=random_state)

        if st.button("Обучить модель"):
            with st.spinner("Обучение модели..."):
                try:
                    model.fit(X_train, y_train)

                    st.session_state['model'] = model
                    st.session_state['model_trained'] = True
                    st.success("Модель успешно обучена!")

                    # Оценка модели
                    st.header("5. Оценка модели")
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                    accuracy = accuracy_score(y_test, y_pred)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred_proba)

                    st.subheader("Метрики")
                    st.write(f"Accuracy: {accuracy:.4f}")
                    st.write(f"ROC-AUC: {roc_auc:.4f}")

                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Предсказанные значения')
                    ax.set_ylabel('Фактические значения')
                    st.pyplot(fig)

                    # Classification Report
                    st.subheader("Classification Report")
                    st.text(class_report)

                    # ROC Curve
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f'{model_choice} (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ошибка при обучении модели: {e}")

        # Предсказание на новых данных
        st.header("6. Предсказание на новых данных")
        with st.form("prediction_form"):
            st.write("Введите значения признаков:")
            col1, col2 = st.columns(2)

            with col1:
                type_ = st.selectbox("Тип продукта", ["L", "M", "H"])
                air_temp = st.number_input("Температура окружающей среды K", value=300.0)
                process_temp = st.number_input("Рабочая температура K", value=310.0)

            with col2:
                rotational_speed = st.number_input("Скорость вращения rpm", value=1500)
                torque = st.number_input("Крутящий момент Nm", value=40.0)
                tool_wear = st.number_input("Износ инструмента min", value=0)

            submit_button = st.form_submit_button("Сделать предсказание")

            if submit_button:
                if not st.session_state.get('model_trained', False):
                    st.error("Сначала обучите модель, нажав кнопку 'Обучить модель'!")
                else:
                    try:
                        # Преобразование введенных данных
                        type_mapping = {"L": 0, "M": 1, "H": 2}
                        input_data = pd.DataFrame({
                            'Type': [type_mapping[type_]],
                            'Air temperature K': [air_temp],
                            'Process temperature K': [process_temp],
                            'Rotational speed rpm': [rotational_speed],
                            'Torque Nm': [torque],
                            'Tool wear min': [tool_wear]
                        })

                        # Масштабирование
                        if st.session_state.get('scaler_fitted', False):
                            input_data[numerical_features] = st.session_state['scaler'].transform(input_data[numerical_features])

                        # Предсказание
                        model = st.session_state['model']
                        prediction = model.predict(input_data)
                        prediction_proba = model.predict_proba(input_data)[:, 1]

                        st.subheader("Результаты предсказания")
                        st.write(f"Предсказанный класс: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                        st.write(f"Вероятность отказа: {prediction_proba[0]:.4f}")

                        fig, ax = plt.subplots()
                        ax.bar(['Нет отказа', 'Отказ'],
                               [1 - prediction_proba[0], prediction_proba[0]],
                               color=['green', 'red'])
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Вероятность')
                        ax.set_title('Вероятность отказа оборудования')
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Ошибка при предсказании: {e}")