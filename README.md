# 📊 Telecom X - Parte 2: Predição de Churn com Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green.svg)
![Status](https://img.shields.io/badge/Status-Concluído-success.svg)

*Projeto de Machine Learning para previsão de evasão de clientes na Telecom X*

</div>

---

## 🎯 **Propósito da Análise**

Este projeto tem como **objetivo principal** desenvolver modelos preditivos capazes de **prever quais clientes têm maior chance de cancelar seus serviços** (churn) na empresa Telecom X. 

A análise utiliza técnicas avançadas de Machine Learning para identificar padrões nos dados dos clientes e antecipar comportamentos de evasão, permitindo que a empresa implemente estratégias proativas de retenção.

### **Problema de Negócio**
- **Taxa de Churn Atual**: 26.58% (1 em cada 4 clientes)
- **Impacto**: Perda significativa de receita recorrente
- **Solução**: Modelo preditivo para identificação precoce de clientes em risco

---

## 📁 **Estrutura do Projeto**

```
challenge-telecom-ml-parte2/
│
├── 📓 analise_churn_ml.ipynb              # Notebook principal com análise completa
├── 📊 dados_tratados.csv                  # Dataset limpo e processado
├── 📈 predicoes_teste.csv                 # Predições do modelo no conjunto de teste
├── 🤖 modelo_churn_gradient_boosting.pkl  # Modelo treinado (melhor performance)
├── ⚙️ scaler_dados.pkl                    # Scaler para normalização dos dados
├── 🏷️ label_encoders.pkl                  # Encoders para variáveis categóricas
├── 📖 README.md                           # Documentação do projeto
└── 📄 LICENSE                             # Licença MIT
```

### **Organização dos Arquivos**

- **`analise_churn_ml.ipynb`**: Notebook Jupyter contendo toda a análise, desde ETL até conclusões estratégicas
- **`dados_tratados.csv`**: Dataset final após limpeza e transformações
- **`modelo_churn_gradient_boosting.pkl`**: Modelo final serializado, pronto para produção
- **`predicoes_teste.csv`**: Resultados das predições com probabilidades de churn
- **Arquivos de suporte**: Scalers e encoders necessários para novas predições

---

## 🔧 **Processo de Preparação dos Dados**

### **1. Extração, Transformação e Limpeza (ETL)**
- **Fonte**: API com dados em formato JSON aninhado
- **Registros processados**: 7.032 clientes (após limpeza)
- **Limpezas realizadas**:
  - Conversão de tipos de dados (cobrança total: texto → numérico)
  - Remoção de registros com churn inválido
  - Tratamento de valores nulos

### **2. Classificação das Variáveis**

#### **Variáveis Categóricas (15)**
- **Demografia**: Gênero, Senior Citizen, Partner, Dependents
- **Serviços**: Phone Service, Multiple Lines, Internet Service
- **Serviços Digitais**: Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies
- **Conta**: Contract, Paperless Billing, Payment Method

#### **Variáveis Numéricas (4)**
- **Comportamentais**: Customer Tenure (tempo como cliente)
- **Financeiras**: Monthly Charges, Total Charges
- **Demográficas**: Senior Citizen (binária)

### **3. Encoding e Normalização**

#### **Label Encoding**
- **Variável Target**: Churn (No=0, Yes=1)
- **Variáveis Categóricas**: Encoding sistemático para todas as 15 variáveis
- **Tratamento especial**: Variáveis com "No service" mantiveram categoria específica

#### **Normalização**
- **Método**: StandardScaler (média=0, desvio=1)
- **Aplicação**: Dados normalizados para Regressão Logística
- **Preservação**: Dados originais mantidos para modelos baseados em árvore

### **4. Divisão dos Dados**
- **Treino**: 5.625 registros (80%) - Estratificado por classe
- **Teste**: 1.407 registros (20%) - Distribuição balanceada
- **Seed**: 42 (para reprodutibilidade)

---

## 🤖 **Modelagem e Justificativas**

### **Algoritmos Selecionados**

| Modelo | Justificativa | Tipo de Dados |
|--------|---------------|---------------|
| **Regressão Logística** | Baseline interpretável, coeficientes claros | Normalizados |
| **Random Forest** | Robusto, lida bem com não-linearidade | Originais |
| **Gradient Boosting** | Alta performance, boa generalização | Originais |

### **Métricas de Avaliação**

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|---------|----------|---------|
| **Gradient Boosting** | **80.74%** | **68.59%** | 50.80% | **58.37%** | **85.10%** |
| Logistic Regression | 79.82% | 65.20% | **51.60%** | 57.61% | 84.44% |
| Random Forest | 79.18% | 63.82% | 50.00% | 56.07% | 83.13% |

**🏆 Melhor Modelo**: Gradient Boosting (ROC-AUC: 85.10%)

---

## 📊 **Principais Insights da Análise Exploratória**

### **🔍 Correlações Mais Fortes com Churn**

1. **Tipo de Contrato** (-0.396): Contratos mensais = maior churn
2. **Tempo como Cliente** (-0.354): Clientes novos = maior risco
3. **Segurança Online** (-0.289): Sem segurança = maior churn
4. **Suporte Técnico** (-0.282): Sem suporte = maior churn
5. **Cobrança Mensal** (+0.193): Maior valor = maior churn

### **📈 Variáveis Mais Importantes (Modelo Final)**

| Posição | Variável | Importância | Insight |
|---------|----------|-------------|---------|
| 1º | **Tipo de Contrato** | 19.98% | Contratos mensais têm 3x mais churn |
| 2º | **Tempo como Cliente** | 19.05% | Primeiros 6 meses são críticos |
| 3º | **Cobrança Mensal** | 14.90% | Clientes sensíveis ao preço |
| 4º | **Cobrança Total** | 12.53% | Histórico de pagamento importa |
| 5º | **Suporte Técnico** | 5.95% | Qualidade do suporte impacta retenção |

### **📊 Gráficos e Visualizações Principais**

1. **Distribuição do Churn**: Gráfico de barras mostrando desbalanceamento (73.4% vs 26.6%)
2. **Matriz de Correlação**: Heatmap das 12 variáveis mais correlacionadas
3. **Curvas ROC**: Comparação de performance entre os 3 modelos
4. **Importância das Features**: Gráficos de barras por modelo
5. **Matriz de Confusão**: Análise detalhada de acertos/erros do melhor modelo

---

## 🚀 **Instruções para Execução**

### **Pré-requisitos**
```bash
Python 3.11+
Jupyter Notebook ou VS Code
```

### **Bibliotecas Necessárias**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests joblib
```

### **Como Executar**

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/AndersonPM15/challenge-telecom-ml-parte2.git
   cd challenge-telecom-ml-parte2
   ```

2. **Abra o notebook**:
   ```bash
   jupyter notebook analise_churn_ml.ipynb
   ```
   ou no VS Code com extensão Python/Jupyter

3. **Execute as células sequencialmente**:
   - **Célula 1**: ETL e carregamento dos dados
   - **Células 2-4**: Análise exploratória e correlações
   - **Células 5-7**: Modelagem e avaliação
   - **Célula 8**: Análise de importância
   - **Célula 9**: Conclusões estratégicas
   - **Célula 10**: Salvamento dos artefatos

### **Carregando Dados Tratados**
```python
import pandas as pd
import joblib

# Carregar dados limpos
df = pd.read_csv('dados_tratados.csv')

# Carregar modelo treinado
model = joblib.load('modelo_churn_gradient_boosting.pkl')

# Carregar preprocessadores
scaler = joblib.load('scaler_dados.pkl')
label_encoders = joblib.load('label_encoders.pkl')
```

---

## 🎯 **Resultados e Recomendações Estratégicas**

### **📊 Principais Descobertas**

- **Taxa de Churn**: 26.58% (crítica para o negócio)
- **Fator #1 de Risco**: Contratos mensais (20% de importância)
- **Janela Crítica**: Primeiros 6 meses de relacionamento
- **Sensibilidade ao Preço**: Cobranças altas correlacionam com churn

### **🚀 Recomendações Prioritárias**

#### **ALTA PRIORIDADE**
1. **Campanha "Contratos Anuais"**: Migração com desconto (Meta: -30% churn)
2. **Programa "Primeiros 100 Dias"**: Onboarding especial (Meta: -40% churn novos clientes)
3. **Revisão de Precificação**: Planos intermediários e ofertas personalizadas

#### **MÉDIA PRIORIDADE**
4. **Melhoria do Suporte Técnico**: Treinamento e chatbot
5. **Programa de Fidelidade**: Benefícios baseados em tenure e CLV

### **📈 Implementação do Modelo**

- **Deploy**: Scoring diário de todos os clientes
- **Segmentação**: Alto (>70%), Médio (30-70%), Baixo Risco (<30%)
- **Ações**: Retenção imediata, campanhas automatizadas, monitoramento passivo
- **Meta**: Reduzir churn de 26.58% para 20% em 12 meses

---

## 📈 **Próximos Passos**

- [ ] **Validação A/B** das estratégias de retenção
- [ ] **Análise de CLV** (Customer Lifetime Value)
- [ ] **Feedback Loop** com resultados das ações
- [ ] **Expansão do Modelo** com novas variáveis (satisfação, uso)
- [ ] **Dashboard Executivo** para monitoramento contínuo

---

## 🛠️ **Tecnologias Utilizadas**

- **Python 3.11+**: Linguagem principal
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Machine Learning
- **Matplotlib/Seaborn**: Visualizações
- **Jupyter Notebook**: Ambiente de desenvolvimento
- **Git/GitHub**: Versionamento e colaboração

---

## 👨‍💻 **Autor**

**Anderson PM** - *Analista de Machine Learning Júnior*
- GitHub: [@AndersonPM15](https://github.com/AndersonPM15)
- LinkedIn: [Anderson Machado](https://www.linkedin.com/in/apm15/)

---

## 📄 **Licença**

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🎉 **Agradecimentos**

- **Alura + Oracle**: Programa ONE - Desafio Data Science
- **Telecom X**: Dados fictícios baseados em casos reais
- **Comunidade Open Source**: Bibliotecas utilizadas

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela no repositório! ⭐**

*Desenvolvido com ❤️ e ☕ para ajudar a Telecom X a reter mais clientes*

</div>
