# main.py
from analysis.temperature_analysis import analyze_temp_demand_relationship, get_temperature_demand_statistics
from data.data_source import EnergyDataSource
from data.mongodb_connector import EnergyDatabaseConnector
from models.prophet_model import EnergyProphetModel
import os
import sys
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("energy_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnergyAnalysis")

# Import our modules


def load_config(config_path):
    """Load configuration from JSON file."""
    default_config = {
        'data_source': {
            'type': 'csv',
            'csv_path': 'data/energy_data.csv',
            'use_real_time': False
        },
        'database': {
            'use_mongodb': False,
            'connection_string': 'mongodb://localhost:27017/'
        },
        'analysis': {
            'forecast_periods': 365,
            'save_figures': True,
            'figures_directory': 'figures'
        }
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)

            # Update default config with user values
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d:
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            config = update_dict(default_config, user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            config = default_config
    else:
        config = default_config

    return config


def save_figures(figures, base_dir="figures"):
    """Save matplotlib figures to directory."""
    # Create figures directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = f"{base_dir}/analysis_{timestamp}"
    os.makedirs(directory, exist_ok=True)

    logger.info(f"Saving figures to {directory}")

    for i, (fig, title) in enumerate(figures):
        filename = f"{directory}/{title.replace(' ', '_').lower()}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved figure {i+1}: {filename}")


def run_analysis(config):
    """Run the energy consumption analysis pipeline."""
    logger.info("Starting energy consumption analysis")

    # Initialize database connector
    db_config = config['database']
    db_connector = EnergyDatabaseConnector(
        connection_string=db_config['connection_string'] if db_config['use_mongodb'] else None
    )

    # Initialize data source
    data_source_config = config['data_source']

    # Load data
    if data_source_config['type'] == 'csv':
        csv_path = data_source_config['csv_path']

        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return

        logger.info(f"Loading data from CSV: {csv_path}")

        # Either load directly or through database
        if db_config['use_mongodb']:
            df = db_connector.load_from_csv(csv_path)
        else:
            df = pd.read_csv(csv_path)

            # Convert timestamp/ds to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])

            # Add temporal features if needed
            if 'ds' in df.columns and 'timestamp' not in df.columns:
                df['timestamp'] = df['ds']

            if 'timestamp' in df.columns:
                if 'hour' not in df.columns:
                    df['hour'] = df['timestamp'].dt.hour
                if 'day_of_week' not in df.columns:
                    df['day_of_week'] = df['timestamp'].dt.dayofweek
                if 'month' not in df.columns:
                    df['month'] = df['timestamp'].dt.month

        logger.info(f"Loaded {len(df)} records")

    else:
        logger.error(
            f"Unsupported data source type: {data_source_config['type']}")
        return

    # Initialize Prophet model
    prophet_model = EnergyProphetModel()

    # Train Prophet model
    logger.info("Training Prophet model")
    model = prophet_model.train(df)

    # Generate forecast
    forecast_periods = config['analysis']['forecast_periods']
    logger.info(f"Generating forecast for {forecast_periods} periods")
    forecast = prophet_model.predict(periods=forecast_periods)

    # Create figures
    figures = []

    # Forecast plot
    logger.info("Creating forecast plot")
    fig_forecast = prophet_model.plot_forecast(forecast)
    figures.append((fig_forecast, "Forecast"))

    # Components plot
    logger.info("Creating components plot")
    fig_components = prophet_model.plot_components(forecast)
    figures.append((fig_components, "Components"))

    # Temperature-demand analysis
    if 'temperature' in df.columns and 'demand' in df.columns:
        logger.info("Analyzing temperature-demand relationship")
        fig_temp_demand = analyze_temp_demand_relationship(df)
        figures.append((fig_temp_demand, "Temperature_Demand_Analysis"))

        # Get temperature-demand statistics
        temp_demand_stats = get_temperature_demand_statistics(df)
        logger.info(
            f"Temperature-demand correlation: {temp_demand_stats['overall_correlation']:.3f}")

    # Save figures
    if config['analysis']['save_figures']:
        save_figures(figures, base_dir=config['analysis']['figures_directory'])

    logger.info("Analysis completed successfully")
    return figures


def run_real_time_analysis(config):
    """Run real-time energy consumption analysis."""
    logger.info("Starting real-time energy consumption analysis")

    # Initialize components
    db_connector = EnergyDatabaseConnector(
        connection_string=config['database']['connection_string'] if config['database']['use_mongodb'] else None
    )

    prophet_model = EnergyProphetModel()

    # Initialize data source
    # You could create a separate config for data_source
    data_source = EnergyDataSource(config_path=None)

    # Load historical data if available
    if config['data_source']['type'] == 'csv' and os.path.exists(config['data_source']['csv_path']):
        historical_df = db_connector.load_from_csv(
            config['data_source']['csv_path'])

        if len(historical_df) > 0:
            logger.info(
                f"Training model with {len(historical_df)} historical records")
            prophet_model.train(historical_df)

    # Define callback for new data
    def process_new_data(data_point):
        # Store in database
        db_connector.store_real_time_data(data_point)

        # Update model (in a real application, you might not update with every point)
        prophet_model.real_time_update(data_point)

        # Log data point
        logger.info(
            f"Processed: {data_point['timestamp']} - Demand: {data_point['demand']:.2f}")

    # Start data source
    data_source.start(callback=process_new_data)

    try:
        logger.info("Real-time analysis running. Press Ctrl+C to stop.")
        while True:
            # In a real application, you might:
            # 1. Periodically retrain the model
            # 2. Generate new forecasts
            # 3. Update visualizations
            # 4. Check for anomalies
            import time
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        data_source.stop()
        db_connector.close_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Consumption Analysis")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--real-time", action="store_true",
                        help="Run in real-time mode")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override real-time setting if specified in command line
    if args.real_time:
        config['data_source']['use_real_time'] = True

    # Run analysis
    if config['data_source']['use_real_time']:
        run_real_time_analysis(config)
    else:
        run_analysis(config)
