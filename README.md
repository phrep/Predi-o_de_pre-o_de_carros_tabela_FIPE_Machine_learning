🎖 Principais etapas do Projeto:

1. Importação de bibliotecas e dataset
2. Visualização e Compreensão dos Dados
3. Limpeza e Tratamento dos dados
4. Análise Exploratória de Dados (EDA)
·   Visualização de dados com gráficos mais complexos (histogramas, boxplots, lineplot, heatmaps).
·  Identificação de padrões, tendências e possíveis outliers.
·  Verificação de correlações e análise de multicolinearidade.
5. Pré-processamento dos Dados
StandardScaler, para padronização de variáveis numéricas.
OneHotEncoder, para codificação de variáveis categóricas em representações binárias.
6. Machine learning
Seleção de modelos adequados.
Treinamento dos modelos no conjunto de treino.
Avaliação inicial dos modelos no conjunto de teste (métricas de desempenho).
Validação e escolha do modelo de machine learning
7. Construção do Pipeline de Modelagem
Configuração do pipeline com os transformadores e o modelo de machine learning
8. Deploy em Produção com novos Dados Reais
 ·  Nesta etapa fiz a seleção de 4 carros polulares e rodei a previsão de valores para agosto/2025
·   Criei um gráfico evolutivo com a predição dos valores + valores dos últimos anos para cada carro

🧠 Considerações finais / Insights:

💡 Maior agrupamento de dados proximo a motores 2.0

💡 Maior distribuição dos dados entre carros modelos 2000 - 2015

💡 Dataset contém outliers, após investigar "avg_price_brl" constatei que são modelos de carros de luxo e decidi manter o dataset, pois a exclusão poderá prejudicar o modelo de realizar previsões para carros de luxo.

💡 Analisando Marcas por quantidades, vemos que os marcas de populares lideram o ranking

💡 Analisando Marcas pela preço médio, vemos que marcas de carros de luxo lideram o ranking

💡 Analisando Ano do veículos pelo preço médio, vemos que o valor aumenta proporcional ao ano do carro 82% dos carros são a gasolina.

💡 Analisando preço de carros por tipo de combustível, vemos que temos quantidade menor quantidade de carros à diesel porem em soma de valores é equivalente aos carros à gasolina

💡 Analisando carros por tipo de cambio, vemos que temos quantidade menor quantidade de carros automáticos, porem em valores os carros de cambio automático são aproximadamente 1/3 mais caros do que manual.

💡 Correlações moderadas, acima de 0.30, entre as colunas: year_model,engine_size e avg_price; indica uma relação positiva moderada entre as variáveis, onde o aumento em uma variável tende a estar associado ao aumento na outra.

💡 Carros mais potentes são mais caros.

💡 Carros mais novos são mais caros.


