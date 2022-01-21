import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

df = pd.read_csv('btc.csv')
df=df.dropna(how='all')
df = df.loc[df['time']>="2017-1-1"]


df['ds']=df["time"]
df['y']=df["PriceUSD"]

## Basicamente como não tive tempo nos ultimos 3 dias
## para me dedicar ao algoritimo, resolvi utilizar a biblioteca disponibilizada
## pelo Facebook 'FBProphet', que basicamente calcula o valor futuro de um ativo

model = Prophet(daily_seasonality=True,)
model.fit(df)
future = model.make_future_dataframe(periods= 9)
prediction = model.predict(future)
fig1 = model.plot(prediction)
fig2 = model.plot_components(prediction)


df['ema9'] = df["y"].ewm(span=9).mean()


print("O ultimo preço de fechamento é:", df['y'].iloc[-2],"USD")
print("o preço alvo é (via FBProphet):", prediction['yhat'].iloc[-1],"USD")
print("A media móvel 9 periodos é :", df['ema9'].iloc[1],"USD")