# real_time_processor.py
from models.prophet_model import EnergyProphetModel
from data.data_source import EnergyDataSource
from data.mongodb_connector import EnergyDatabaseConnector
import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("energy_realtime.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnergyRealTime")

# Import our modules


def load_config(config_path):
    """Load configuration from JSON file."""
    default_config = {
        "data_source": {
            "type": "real-time",
            "use_real_time": True,
            "real_time_config": {
                "source_type": "simulation",
                "simulation": {
                    "interval_seconds": 5.0
                }
            }
        },
        "database": {
            "use_mongodb": true,
            "connection_string": "mongodb+srv://nguyendangvuduy12t2:Dudikun13@@energy-consumption-bda.obgdd.mongodb.net/?retryWrites=true&w=majority&appName=energy-consumption-bda",
            "db_name": "energy_consumption_bda",
            "collection_name": "consumption_data"
        },
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


def run_real_time_processor(config):
    """Run real-time data collection and storage."""
    logger.info("Starting real-time energy data processor")

    # Initialize database connector
    db_config = config['database']
    db_connector = EnergyDatabaseConnector(
        db_name=db_config.get('db_name', 'energy_consumption_bda')
    )

    # Check database connection
    if not db_connector.is_connected():
        logger.error(
            "Failed to connect to MongoDB. Please check your configuration and ensure MongoDB is running.")
        return

    logger.info("Successfully connected to MongoDB")

    # Initialize data source
    data_source_config = config['data_source']['real_time_config']
    data_source = EnergyDataSource(config_path=None)
    data_source.config = data_source_config  # Override with our config

    # Setup stats for reporting
    start_time = datetime.now()
    data_points_collected = 0

    # Define callback for new data
    def process_data_point(data_point):
        nonlocal data_points_collected

        # Store in MongoDB
        db_connector.store_real_time_data(data_point)

        # Increment counter
        data_points_collected += 1

        # Log every 10th data point to avoid excessive logging
        if data_points_collected % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = data_points_collected / elapsed if elapsed > 0 else 0
            logger.info(
                f"Collected {data_points_collected} data points ({rate:.2f} points/sec)")
            logger.info(
                f"Latest: {data_point['timestamp']} - Demand: {data_point['demand']:.2f}, Temp: {data_point['temperature']:.1f}°C")

    # Start data source
    data_source.start(callback=process_data_point)

    try:
        logger.info("Real-time processor running. Press Ctrl+C to stop.")

        # Keep running until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        data_source.stop()
        db_connector.close_connection()

        # Print final stats
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Session summary: Collected {data_points_collected} data points in {elapsed:.1f} seconds")
        if elapsed > 0:
            logger.info(
                f"Average collection rate: {data_points_collected / elapsed:.2f} points/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Energy Data Processor")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run processor
    run_real_time_processor(config)
