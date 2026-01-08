# 🛫 Tech Challenge - Machine Learning Engineering (Fase 3)

## Análise Preditiva de Voos Comerciais dos EUA (2015)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Dataset](#-dataset)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Notebooks](#-notebooks)
- [Resultados](#-resultados)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Como Executar](#-como-executar)
- [Conclusões](#-conclusões)
- [Autores](#-autores)

---

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como parte do **Tech Challenge - Fase 3** do curso de **Machine Learning Engineering** da FIAP/Alura. O objetivo é aplicar técnicas de Machine Learning para análise e previsão de padrões no setor de aviação comercial dos Estados Unidos.

### Objetivos

O projeto aborda três frentes principais de modelagem preditiva:

1. **🟢 Classificação**: Prever se um voo terá atraso significativo (>15 minutos)
2. **🔵 Regressão**: Prever a duração do atraso de chegada em minutos
3. **🟠 Clusterização**: Segmentar aeroportos, rotas e companhias aéreas por perfil operacional

---

## 📊 Dataset

Os dados utilizados são provenientes do **U.S. Department of Transportation's Bureau of Transportation Statistics**, contendo informações detalhadas sobre voos comerciais domésticos dos EUA em 2015.

### Arquivos

| Arquivo | Registros | Descrição |
|---------|-----------|-----------|
| `flights.csv` | ~5.8 milhões | Dados completos de voos (origem, destino, horários, atrasos, cancelamentos) |
| `airlines.csv` | 14 | Informações das companhias aéreas |
| `airports.csv` | 322 | Informações dos aeroportos (código, nome, localização) |

### Variáveis Principais

- **Temporais**: `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`, `SCHEDULED_DEPARTURE`, `SCHEDULED_ARRIVAL`
- **Operacionais**: `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `DISTANCE`
- **Atrasos**: `DEPARTURE_DELAY`, `ARRIVAL_DELAY`, `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`
- **Status**: `CANCELLED`, `CANCELLATION_REASON`, `DIVERTED`

---

## 📁 Estrutura do Projeto

```
MLET-Phase-3/
├── 📂 data/                          # Datasets
│   ├── airlines.csv                  # Companhias aéreas
│   ├── airports.csv                  # Aeroportos
│   └── flights.csv                   # Dados de voos (~5.8M registros)
├── 📂 docs/                          # Documentação
│   └── Tech Challenge Fase 3.pdf     # Enunciado do desafio
├── 📂 models/                        # Modelos salvos
│   ├── airlines_clustered.csv        # Resultado clusterização airlines
│   ├── airports_clustered.csv        # Resultado clusterização airports
│   ├── routes_clustered.csv          # Resultado clusterização rotas
│   └── clustering_metadata.json      # Metadados dos modelos
├── 📂 notebooks/                     # Jupyter Notebooks
│   ├── 1 EDA.ipynb                   # Análise Exploratória de Dados
│   ├── 2 Classification Model.ipynb  # Modelo de Classificação
│   ├── 3 Regression Model.ipynb      # Modelo de Regressão
│   └── 4 Clusterization Model.ipynb  # Modelo de Clusterização
├── 📂 envs/                          # Ambiente virtual (conda)
├── 📂 src/                           # Código fonte
│   └── main.py                       # Script principal
├── LICENSE                           # Licença MIT
└── README.md                         # Este arquivo
```

---

## 📓 Notebooks

### 1. Análise Exploratória de Dados (EDA)

**Arquivo**: `notebooks/1 EDA.ipynb`

Análise completa do dataset de voos comerciais, incluindo:

- ✅ Estatísticas descritivas detalhadas
- ✅ Visualizações com insights operacionais
- ✅ Tratamento de valores ausentes
- ✅ Análise de correlações
- ✅ Identificação de padrões temporais (horários, dias, meses)
- ✅ Comparação entre companhias aéreas
- ✅ Propostas de modelagem para as próximas fases

**Principais Insights:**
- Taxa de cancelamento: ~1.5% dos voos
- Atrasos aumentam progressivamente ao longo do dia (efeito cascata)
- Principais causas de atraso: Late Aircraft Delay e Airline Delay
- Principal causa de cancelamento: Condições meteorológicas

---

### 2. Modelo de Classificação

**Arquivo**: `notebooks/2 Classification Model.ipynb`

Pipeline completo de classificação binária para prever atrasos significativos.

**Problema**: Classificar se um voo terá atraso > 15 minutos na chegada

**Modelos Implementados:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

**Técnicas Utilizadas:**
- SMOTE para balanceamento de classes
- StandardScaler para normalização
- Label Encoding para variáveis categóricas
- Stratified K-Fold Cross-Validation

**Resultados:**

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.5764 | 0.2388 | **0.6240** | **0.3454** | 0.6311 |
| Random Forest | 0.7577 | 0.3255 | 0.3291 | 0.3273 | 0.6601 |
| Decision Tree | 0.7336 | 0.2637 | 0.2719 | 0.2677 | 0.6137 |
| Gradient Boosting | **0.8188** | **0.4640** | 0.0737 | 0.1272 | **0.6724** |

**Melhor Modelo**: Logistic Regression (maior F1-Score e Recall)

---

### 3. Modelo de Regressão

**Arquivo**: `notebooks/3 Regression Model.ipynb`

Pipeline completo de regressão para prever a duração do atraso em minutos.

**Variável Alvo**: `ARRIVAL_DELAY` (contínua, em minutos)

**Modelos Implementados:**
- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

**Features Engineered:**
- Agregações históricas por aeroporto, companhia e hora
- Indicadores de período de feriados
- Volume de operações por origem/destino

**Resultados:**

| Modelo | MAE (min) | RMSE (min) | R² |
|--------|-----------|------------|-----|
| **Linear Regression** | **16.84** | 26.68 | 0.0158 |
| Ridge Regression | 16.84 | 26.68 | 0.0158 |
| Lasso Regression | 16.89 | 26.70 | 0.0151 |
| Random Forest | 17.14 | 27.76 | -0.0659 |
| Gradient Boosting | 17.62 | 27.43 | -0.0396 |
| LightGBM | 16.97 | 26.74 | 0.0120 |
| XGBoost | 17.65 | 27.16 | -0.0203 |

**Melhor Modelo**: Linear Regression (menor MAE)

**Features Mais Importantes:**
1. Distância do voo (17.89%)
2. Atraso médio do aeroporto de destino (13.94%)
3. Atraso médio do aeroporto de origem (12.12%)
4. Atraso médio por hora (10.43%)
5. Volume de voos no aeroporto (9.68%)

---

### 4. Modelo de Clusterização

**Arquivo**: `notebooks/4 Clusterization Model.ipynb`

Pipeline completo de clusterização para segmentação operacional.

**Entidades Analisadas:**
- Aeroportos (930)
- Rotas (5.641)
- Companhias Aéreas (14)

**Algoritmos Implementados:**
- K-Means
- Gaussian Mixture Model (GMM)
- Hierarchical Clustering (Agglomerative)
- DBSCAN

**Técnicas de Avaliação:**
- Método do Cotovelo (Elbow Method)
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Dendrogramas

**Resultados:**

| Entidade | Clusters | Silhouette Score | Melhor Algoritmo |
|----------|----------|------------------|------------------|
| **Aeroportos** | 4 | **0.734** | K-Means |
| Rotas | 5 | 0.175 | K-Means |
| Companhias Aéreas | 3 | 0.253 | K-Means |

#### Perfis dos Clusters de Aeroportos:

| Cluster | Qtd | Perfil | Exemplos |
|---------|-----|--------|----------|
| 0 | 884 | Regionais Eficientes | BNA, PDX, STL |
| 1 | 5 | Mega Hubs | ATL, ORD, DFW, DEN, LAX |
| 2 | 24 | Grandes Aeroportos | SFO, PHX, IAH, LAS |
| 3 | 17 | Microaeroportos Problemáticos | - |

#### Perfis dos Clusters de Companhias:

| Cluster | Companhias | Market Share | Atraso Médio | Perfil |
|---------|------------|--------------|--------------|--------|
| 0 | F9, NK | 3.6% | 13.5 min | Low-Cost com Atrasos |
| 1 | AA, DL, UA, WN... | 87.7% | 4.9 min | Major Carriers |
| 2 | AS, HA, VX | 8.8% | 2.4 min | Regionais Premium |

---

## 📈 Resultados Consolidados

### Classificação (Previsão de Atrasos Significativos)
- **Modelo Final**: Logistic Regression
- **F1-Score**: 0.3454
- **Recall**: 62.40% (detecta maioria dos atrasos)
- **Feature mais importante**: Hora do dia (34%)

### Regressão (Previsão de Duração do Atraso)
- **Modelo Final**: Linear Regression
- **MAE**: 16.84 minutos
- **Interpretação**: Erro médio de ~17 minutos na previsão
- **Feature mais importante**: Distância do voo (18%)

### Clusterização (Segmentação Operacional)
- **Aeroportos**: 4 clusters com separação excelente (Silhouette: 0.734)
- **Insight principal**: 5 mega hubs (ATL, ORD, DFW, DEN, LAX) concentram o maior volume

---

## 🛠 Tecnologias Utilizadas

### Linguagem e Ambiente
- Python 3.11+
- Jupyter Notebook
- Conda (gerenciamento de ambiente)

### Manipulação de Dados
- Pandas
- NumPy

### Visualização
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn (SMOTE)

### Clusterização
- SciPy (hierarchical clustering)
- Scikit-learn (K-Means, DBSCAN, GMM)

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.11+
- Conda ou pip
- ~2GB de RAM disponível (dataset grande)

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/lucasedmundo11/MLET-Phase-3.git
cd MLET-Phase-3
```

2. **Crie e ative o ambiente virtual:**
```bash
# Com Conda
conda create -n mlet-phase3 python=3.11
conda activate mlet-phase3

# Ou com venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

3. **Instale as dependências:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn scipy jupyter
```

4. **Execute os notebooks:**
```bash
jupyter notebook notebooks/
```

### Ordem de Execução Recomendada

1. `1 EDA.ipynb` - Análise exploratória
2. `2 Classification Model.ipynb` - Modelo de classificação
3. `3 Regression Model.ipynb` - Modelo de regressão
4. `4 Clusterization Model.ipynb` - Modelo de clusterização

---

## 📝 Conclusões

### Principais Descobertas

1. **Padrão de Efeito Cascata**: Atrasos acumulam ao longo do dia, tornando voos matinais mais pontuais

2. **Previsibilidade Limitada**: Atrasos de voos são inerentemente difíceis de prever (~17 min de erro médio) devido a fatores externos não disponíveis nos dados (clima em tempo real, problemas técnicos aleatórios)

3. **Segmentação Clara de Aeroportos**: Os 5 mega hubs (ATL, ORD, DFW, DEN, LAX) operam em escala completamente diferente dos demais, justificando estratégias operacionais distintas

4. **Trade-off Low-Cost**: Companhias ultra low-cost (Frontier, Spirit) têm os maiores atrasos médios (13.5 min vs 2.4 min das regionais premium)

5. **Features Mais Preditivas**: Hora do dia, distância do voo e histórico de atrasos do aeroporto são os principais fatores preditivos

### Limitações

- Dataset de apenas 2015 (padrões podem ter mudado)
- Ausência de dados meteorológicos em tempo real
- Features agregadas podem conter data leakage
- R² baixo nos modelos de regressão devido à natureza estocástica dos atrasos

### Trabalhos Futuros

- Integrar APIs de clima para enriquecer predições
- Implementar modelos de séries temporais
- Desenvolver pipeline de produção com MLOps
- Adicionar explicabilidade com SHAP values

---

## 👥 Autores

- **Giovanna de Lima** - [GitHub](https://github.com/Badgioo)
- **Lucas Edmundo** - [GitHub](https://github.com/lucasedmundo11)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

*Projeto desenvolvido para o Tech Challenge - Fase 3 | Machine Learning Engineering | FIAP/Alura*