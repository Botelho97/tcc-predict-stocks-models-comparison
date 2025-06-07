# Precificação de Ações na Bolsa Brasileira

**Comparação Entre Modelos de Machine Learning e Séries Temporais**

Este projeto faz parte do Trabalho de Conclusão de Curso do MBA em Data Science e Analytics da USP/Esalq. O objetivo é comparar o desempenho de diferentes abordagens de modelagem para previsão de preços de ações negociadas na B3, utilizando modelos estatísticos e de aprendizado de máquina.

## Objetivo

Avaliar a eficácia de modelos preditivos aplicados à precificação de ações, analisando o desempenho de técnicas estatísticas de séries temporais e modelos supervisionados de machine learning.

## Modelos Utilizados

### Modelos de Séries Temporais:
- Naive
- Mean
- Drift
- Naive Seasonal
- Simple Exponencial Smoothing (SES)
- Holt
- Holt-Winters
- ARIMA (AutoRegressive Integrated Moving Average)
- SARIMA 
- Prophet (Meta/Facebook)

### Modelos de Machine Learning:
- Decision Tree
- Random Forest
- XGBoost (Extreme Gradient Boosting)
- LightGBM
- LSTM (Long Short-Term Memory)

## Metodologia

- **Tipo de Estudo**: Quantitativo e experimental
- **Fontes de Dados**: Yahoo Finance e Banco Central do Brasil
- **Pré-processamento**:
  - Tratamento de valores ausentes
  - Normalização e padronização
  - Cálculo de métricas anuais
  - Feature Engineering

## 📂 Estrutura do Projeto

```
📁 projeto-precificacao-acoes
├── data/              # Dados brutos e tratados
├── data_wrangling/    # Scripts de Data Wrangling
├── src/               # Scripts auxiliares (funções para treinamento e output de modelos)
├── main.py            # Execução principal do projeto
├── README.md
└── LICENSE
```

## 📏 Métricas de Avaliação

- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coeficiente de Determinação)

## 📚 Referências

- Fischer & Krauss (2018). *Deep Learning with LSTM for Financial Market Predictions*.
- Gu et al. (2020). *Empirical Asset Pricing via Machine Learning*.
- Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice*.
- Makridakis et al. (2018). *Statistical and ML Forecasting Methods: Concerns and Ways Forward*.

## 📝 Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).
