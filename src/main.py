import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv('data/energy_data.csv')
df['ds'] = pd.to_datetime(df['ds'])

# Now filter with proper datetime objects (keeping the gap for Prophet)
mask = (df['ds'] > pd.to_datetime('2010-01-01')) & (df['ds'] < pd.to_datetime('2011-01-01'))
df.loc[mask, 'y'] = None

# --- Prophet forecasting (keeping your original code) ---
m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

# --- New code to analyze temperature and demand correlation ---
# Function to analyze temperature-demand relationship
def analyze_temp_demand_relationship(df):
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot of temperature vs demand
    sns.scatterplot(x='temperature', y='demand', data=df, alpha=0.3, ax=axes[0, 0])
    axes[0, 0].set_title('Temperature vs Demand')
    
    # Add regression line
    sns.regplot(x='temperature', y='demand', data=df, scatter=False, 
                line_kws={"color":"red"}, ax=axes[0, 0])
    
    # Calculate and display correlation coefficient
    corr = df['temperature'].corr(df['demand'])
    axes[0, 0].annotate(f'Correlation: {corr:.3f}', 
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=12, ha='left', va='top')
    
    # 2. Temperature vs demand by month (to see seasonal effects)
    monthly_corr = df.groupby('month')[['temperature', 'demand']].corr().iloc[::2, 1].reset_index()
    monthly_corr = monthly_corr.rename(columns={'demand': 'correlation'})
    sns.barplot(x='month', y='correlation', data=monthly_corr, ax=axes[0, 1])
    axes[0, 1].set_title('Temperature-Demand Correlation by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Correlation Coefficient')
    
    # 3. Temperature vs demand by hour (to see daily patterns)
    sns.boxplot(x='hour', y='demand', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Demand Distribution by Hour')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Demand')
    
    # 4. Heatmap of demand by temperature and hour
    pivot = df.pivot_table(values='demand', 
                          index=pd.cut(df['temperature'], bins=10), 
                          columns='hour', 
                          aggfunc='mean')
    sns.heatmap(pivot, cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title('Demand by Temperature Range and Hour')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Temperature Range')
    
    plt.tight_layout()
    return fig

# Create the temperature-demand analysis figure
fig3 = analyze_temp_demand_relationship(df)

def saveFigures(figures):
    directory = 'figure/attempts'
    attempt = 1
    while os.path.exists(f'{directory}-{attempt}'):
        attempt += 1
    directory = f'{directory}-{attempt}'
    os.mkdir(directory)
    
    for i, fig in enumerate(figures):
        fig.savefig(f'{directory}/figure{i+1}.png')

# Save all figures including the new temperature-demand analysis
saveFigures([fig1, fig2, fig3])

# Print summary statistics about temperature-demand relationship
print("Overall correlation between temperature and demand:", 
      df['temperature'].corr(df['demand']))

# Check for non-linear relationship using temperature ranges
temp_ranges = pd.cut(df['temperature'], bins=5)
print("\nAverage demand by temperature range:")
print(df.groupby(temp_ranges)['demand'].mean())