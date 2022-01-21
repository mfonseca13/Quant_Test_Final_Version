##importando bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

##obtendo dados
df = pd.read_csv('btc.csv')

##Removendo dados NaN
df=df.dropna(how='all')

##Reduzindo a amostragem para melhorar representação grafica,
##O funcionamento do algoritimo nao necessita dessa mudança
df = df.loc[df['time']>="2017-1-1"]



## Basicamente como não tive tempo nos ultimos 3 dias
## para me dedicar ao algoritimo, resolvi utilizar a biblioteca disponibilizada
## pelo Facebook 'FBProphet', que basicamente calcula o valor futuro de um ativo

##Para utilizar o Prophet precisamos de 2  colunas sendo elas
## ds(tempo), e y (preço)

df['ds']=df["time"]
df['y']=df["PriceUSD"]

##chamando o prophet, e setando dados a serem executados e modelados
model = Prophet(daily_seasonality=True,)
model.fit(df)

##Modelando o dataframe,escolhendo periodo a ser previsto
future = model.make_future_dataframe(periods= 9)
prediction = model.predict(future)

##Plotando resultados graficamente
fig1 = model.plot(prediction)
fig2 = model.plot_components(prediction)

##Calculando media movel de 9 periodos
df['ema9'] = df["y"].ewm(span=9).mean()

##retornando valores
print("O ultimo preço de fechamento é:", df['y'].iloc[-2],"USD")
print("o preço alvo é (via FBProphet):", prediction['yhat'].iloc[-1],"USD")
print("A media móvel 9 periodos é :", df['ema9'].iloc[1],"USD")
