# Predição de Espécies Iris com Scikit-Learn

## Descrição

Este é um **projeto de aprendizado** onde experimentei com **Pipelines do Scikit-Learn** para treinar e comparar diferentes modelos de classificação no dataset Iris e criar uma interface web usando **Streamlit**.

O objetivo foi entender como:
- Carregar e processar dados com Pandas
- Dividir dados em treino/teste
- Criar Pipelines para padronização
- Treinar múltiplos modelos de classificação
- Salvar e carregar modelos com joblib
- Criar interfaces web com Streamlit

## 📊 Modelos Treinados

1. **LogisticRegression** - Classificação logística
2. **DecisionTreeClassifier** - Árvore de decisão
3. **KNeighborsClassifier** - K-Vizinhos mais próximos
4. **LinearDiscriminantAnalysis** - Análise discriminante linear
5. **SVC** - Máquina de vetores de suporte
6. **GaussianNB** - Naive Bayes Gaussiano

## 🚀 Como Rodar

### 1. Instalar dependências
```bash
pip install pandas scikit-learn joblib streamlit seaborn matplotlib numpy scipy
```

### 2. Treinar os modelos
Execute o script de treinamento:
```bash
python treino.py
```

Isso irá:
- Baixar o dataset Iris
- Dividir em treino (80%) e teste (20%)
- Treinar todos os 6 modelos
- Salvar os modelos em `models/`
- Exibir acurácia e relatório de classificação

### 3. Rodar a interface web
```bash
streamlit run app.py
```

A aplicação abrirá em `http://localhost:8501`

## 📁 Estrutura do Projeto

```
d:\projetos_vscode\test\
├── treino.py           # Script de treinamento dos modelos
├── app.py              # Interface web com Streamlit
├── README.md           # Este arquivo
└── models/             # Pasta com modelos salvos
    ├── predict_iris_model1.joblib
    ├── predict_iris_model2.joblib
    ├── predict_iris_model3.joblib
    ├── predict_iris_model4.joblib
    ├── predict_iris_model5.joblib
    └── predict_iris_model6.joblib
```

## 🎯 Como Usar a Interface Web

1. Selecione um modelo no dropdown
2. Ajuste os valores das medidas da flor (Sepal length, Sepal width, Petal length, Petal width)
3. Clique em **"Prever"**
4. Veja a classe prevista e as probabilidades de cada espécie

## 📚 Dataset

- **Fonte**: UCI Machine Learning Repository
- **Amostras**: 150 flores
- **Classes**: Setosa, Versicolor, Virginica
- **Features**: 4 medidas (em cm)

## 🔧 Tecnologias Usadas

- **Python 3.x**
- **Scikit-Learn** - Modelos de ML
- **Pandas** - Manipulação de dados
- **Streamlit** - Interface web
- **Joblib** - Serialização de modelos
- **Matplotlib & Seaborn** - Visualizações

## ✨ O que Aprendi

- Pipelines padronizam o fluxo de ML
- Diferentes modelos têm diferentes desempenhos
- Validação de dados é crucial
- Streamlit torna fácil criar interfaces para modelos ML

---

**Nota**: Este é um projeto educacional para praticar conceitos de Machine Learning.