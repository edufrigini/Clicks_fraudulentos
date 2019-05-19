# DSA - DATA SCIENCE ACADEMY 
# FORMACAO CIENTISTA DE DADOS
# LIGUAGEM R COM AZURE MACHINE LEARNING
#
# PROJETO 1, PREVISAO DE CLICKS FRAUDULENTOS
# ALUNO: EDUARDO FRIGINI DE JESUS 
# 
setwd("C:/FCD/BigDataRAzure/Projeto1")
getwd()

# carregando as bibliotecas, se nao estiver instalada, instalar install.packages("nome do pacote")
library(data.table)
library(lubridate)
library(DMwR)
library(dplyr)
library(tidyr)
library(ggplot2)
library(randomForest)
library("ROCR")
library(caret)

# Carregando os dados na memoria
# Usando o arquivo train.csv para treinar o modelo para producao
dados <- fread("train.csv", header = T, stringsAsFactors = F )
head(dados)
str(dados)

## Convertendo as variáveis para o tipo fator (categórica)
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

# Variáveis do tipo fator
# nao converti as outras pq eram mts categorias e a floresta randomica nao processa mts categorias
dados <- to.factors(df = dados, variables = 'is_attributed')
str(dados)
head(dados)

# Verificar como esta a distribuicao dos dados
table(dados$is_attributed)

# Data set desbalanceado para o treino
#     0         1 
# 184447044    456846 

# Balancear o número de casos positivos e negativos
click     <- dados[is_attributed==1,]
no_click  <- dados[sample(1:nrow(dados), 456846, replace = F)]

# juntando os clicks com a amostra dos nao clicks
dados_ok <- bind_rows(click, no_click, id = NULL)

# Verificando novamente se os dados estao balanceados
table(dados_ok$is_attributed)
# Data set balanceado
#   0      1 
# 455750 457942 

# Gerando dados de treino e de teste
sample <- sample.int(n = nrow(dados_ok), size = floor(.7*nrow(dados_ok)), replace = F)
treino <- dados_ok[sample, ]
teste  <- dados_ok[-sample, ]

# Verificando o numero de linhas
nrow(treino)
nrow(teste)

# Feature Selection
modelo <- randomForest(is_attributed ~ app 
                        + device 
                        + os 
                        + channel
                        + ip,
                        data = treino, 
                        ntree = 100, nodesize = 10, importance = T)

varImpPlot(modelo)
# Importancia de cada variavel
# ip 118
# app 80
# channel 45
# os 22
# device 7

print(modelo)
# erro rate: 7.99%

# Gerando previsões nos dados de teste
previsoes <- data.frame(observado = teste$is_attributed,
                        previsto = predict(modelo, newdata = teste))


# Visualizando o resultado
View(previsoes)
View(teste)

# Gerando as classes de dados
class1 <- predict(modelo, newdata = teste, type = 'prob')
class2 <- teste$is_attributed

# Gerando a curva ROC
pred <- prediction(class1[,2], class2)
perf <- performance(pred, "tpr","fpr") 
plot(perf, col = rainbow(10))

# Gerando Confusion Matrix com o Caret
confusionMatrix(previsoes$observado, previsoes$previsto)
# Accuracy = 0.92


