from pathlib import Path
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_OPTIONS = {
    "1 - LogisticRegression": MODEL_DIR / "predict_iris_model1.joblib",
    "2 - DecisionTreeClassifier": MODEL_DIR / "predict_iris_model2.joblib",
    "3 - KNeighborsClassifier": MODEL_DIR / "predict_iris_model3.joblib",
    "4 - LinearDiscriminantAnalysis": MODEL_DIR / "predict_iris_model4.joblib",
    "5 - SVC": MODEL_DIR / "predict_iris_model5.joblib",
    "6 - GaussianNB": MODEL_DIR / "predict_iris_model6.joblib",
}

@st.cache_resource
def carregar_modelo(model_path: str):
    return joblib.load(model_path)

st.set_page_config(page_title="Predição Iris", page_icon="🌸")
st.title("🌸 Predição de espécie Iris")
st.write("Escolha um modelo treinado e informe as medidas da flor.")

modelo_nome = st.selectbox("Selecione o modelo", list(MODEL_OPTIONS.keys()))
model_path = MODEL_OPTIONS[modelo_nome]

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal length", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal width", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
with col2:
    petal_length = st.number_input("Petal length", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal width", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Prever"):
    if not model_path.exists():
        st.error(f"Modelo não encontrado: {model_path}")
    else:
        modelo = carregar_modelo(str(model_path))
        amostra = [[sepal_length, sepal_width, petal_length, petal_width]]

        pred = modelo.predict(amostra)[0]

        st.subheader("Resultado")
        st.success(f"Classe prevista: **{pred}**")
        st.caption(f"Modelo usado: {modelo_nome}")

        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(amostra)[0]
            classes = modelo.classes_
            distribuicao = {str(c): float(p) for c, p in zip(classes, proba)}
            st.write("Probabilidades:")
            st.json(distribuicao)