import pandas as pd
from prophet import Prophet
import os

df = pd.read_csv('data/energy_data.csv')
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


## Save the figures
def saveFigures(figures):
    directory = 'figure/attempts'
    attempt = 1
    while os.path.exists(f'{directory}-{attempt}'):
        attempt += 1
    directory = f'{directory}-{attempt}'
    os.mkdir(directory)
    
    for i, fig in enumerate(figures):
        fig.savefig(f'{directory}/figure{i+1}.png')

saveFigures([fig1, fig2])