# analysis/temperature_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TemperatureAnalysis")


def analyze_temp_demand_relationship(df):
    """
    Analyze the relationship between temperature and energy demand.

    Args:
        df: DataFrame with 'temperature' and 'demand' columns

    Returns:
        Matplotlib figure with analysis plots
    """
    logger.info("Analyzing temperature-demand relationship")

    # Ensure required columns exist
    required_columns = ['temperature', 'demand']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            raise ValueError(f"DataFrame must contain '{col}' column")

    # Add datetime components if needed
    if 'ds' in df.columns:
        df['date'] = pd.to_datetime(df['ds'])
    elif 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'])

    if 'date' in df.columns:
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
    else:
        logger.warning(
            "No datetime column found. Some analyses will be limited.")

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Scatter plot of temperature vs demand
    sns.scatterplot(x='temperature', y='demand',
                    data=df, alpha=0.3, ax=axes[0, 0])
    axes[0, 0].set_title('Temperature vs Demand')

    # Add regression line
    sns.regplot(x='temperature', y='demand', data=df, scatter=False,
                line_kws={"color": "red"}, ax=axes[0, 0])

    # Calculate and display correlation coefficient
    corr = df['temperature'].corr(df['demand'])
    axes[0, 0].annotate(f'Correlation: {corr:.3f}',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=12, ha='left', va='top')

    # 2. Temperature vs demand by month (to see seasonal effects)
    if 'month' in df.columns:
        try:
            monthly_corr = df.groupby(
                'month')[['temperature', 'demand']].corr().iloc[::2, 1].reset_index()
            monthly_corr = monthly_corr.rename(
                columns={'demand': 'correlation'})
            sns.barplot(x='month', y='correlation',
                        data=monthly_corr, ax=axes[0, 1])
            axes[0, 1].set_title('Temperature-Demand Correlation by Month')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Correlation Coefficient')
        except Exception as e:
            logger.error(f"Error in monthly correlation plot: {e}")
            axes[0, 1].set_title('Monthly Correlation Error')
            axes[0, 1].text(0.5, 0.5, str(e), ha='center', va='center')
    else:
        axes[0, 1].set_title('Monthly Correlation (No Month Data)')

    # 3. Boxplot by hour
    if 'hour' in df.columns:
        try:
            sns.boxplot(x='hour', y='demand', data=df, ax=axes[1, 0])
            axes[1, 0].set_title('Demand Distribution by Hour')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Demand')
        except Exception as e:
            logger.error(f"Error in hourly boxplot: {e}")
            axes[1, 0].set_title('Hourly Distribution Error')
            axes[1, 0].text(0.5, 0.5, str(e), ha='center', va='center')
    else:
        axes[1, 0].set_title('Hourly Distribution (No Hour Data)')

    # 4. Heatmap of demand by temperature and hour
    if 'hour' in df.columns:
        try:
            # Create temperature bins
            df['temp_bin'] = pd.cut(df['temperature'], bins=10)

            pivot = df.pivot_table(
                values='demand',
                index='temp_bin',
                columns='hour',
                aggfunc='mean'
            )

            sns.heatmap(pivot, cmap='YlOrRd', ax=axes[1, 1])
            axes[1, 1].set_title('Demand by Temperature Range and Hour')
            axes[1, 1].set_xlabel('Hour of Day')
            axes[1, 1].set_ylabel('Temperature Range')
        except Exception as e:
            logger.error(f"Error in temperature-hour heatmap: {e}")
            axes[1, 1].set_title('Heatmap Error')
            axes[1, 1].text(0.5, 0.5, str(e), ha='center', va='center')
    else:
        axes[1, 1].set_title('Temperature-Hour Heatmap (No Hour Data)')

    plt.tight_layout()
    logger.info("Temperature-demand analysis completed")

    return fig


def get_temperature_demand_statistics(df):
    """
    Calculate statistical measures of the temperature-demand relationship.

    Args:
        df: DataFrame with 'temperature' and 'demand' columns

    Returns:
        Dictionary with statistics
    """
    stats = {}

    # Overall correlation
    stats['overall_correlation'] = df['temperature'].corr(df['demand'])

    # Check for non-linear relationship using temperature ranges
    temp_ranges = pd.cut(df['temperature'], bins=5)
    stats['demand_by_temp_range'] = df.groupby(
        temp_ranges)['demand'].mean().to_dict()

    # Calculate temperature thresholds
    stats['temp_min'] = df['temperature'].min()
    stats['temp_max'] = df['temperature'].max()
    stats['temp_mean'] = df['temperature'].mean()

    # Calculate demand thresholds
    stats['demand_min'] = df['demand'].min()
    stats['demand_max'] = df['demand'].max()
    stats['demand_mean'] = df['demand'].mean()
    stats['demand_std'] = df['demand'].std()

    # Peak demand temperature
    peak_demand_temp = df.loc[df['demand'].idxmax(), 'temperature']
    stats['peak_demand_temperature'] = peak_demand_temp

    # Temperature at minimum demand
    min_demand_temp = df.loc[df['demand'].idxmin(), 'temperature']
    stats['min_demand_temperature'] = min_demand_temp

    return stats
