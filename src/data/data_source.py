# data/data_source.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import os
import json
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnergyDataSource")


class EnergyDataSource:
    """Data source for energy consumption data."""

    def __init__(self, config_path=None):
        """Initialize data source.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize internal state
        self.running = False
        self.thread = None
        self.callback = None

        # Current simulated day conditions (if using simulation)
        self._reset_simulation_state()

    def _load_config(self, config_path):
        """Load configuration from file."""
        default_config = {
            'source_type': 'simulation',  # 'simulation', 'api', or 'csv'
            'simulation': {
                'interval_seconds': 5.0,
                'base_demand': 50.0,
                'base_temperature': 20.0,
                'day_cycle_amplitude': 15.0,  # Temperature variation during day
                'season_cycle_amplitude': 15.0,  # Temperature variation during year
                'weekday_factor': 10.0,  # Higher demand on weekdays
                'hour_factor': 15.0,  # Demand variation during day
                'noise_factor': 5.0,  # Random noise in demand
                'temperature_noise': 2.0  # Random noise in temperature
            },
            'api': {
                'url': None,
                'api_key': None,
                'interval_seconds': 60.0
            },
            'csv': {
                'path': 'data/energy_data.csv',
                'interval_seconds': 5.0,
                'loop': True  # Whether to loop through the CSV repeatedly
            }
        }

        # If config path provided, load and merge with defaults
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

                default_config = update_dict(default_config, user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        return default_config

    def _reset_simulation_state(self):
        """Reset simulation state variables."""
        self.csv_data = None
        self.csv_index = 0
        self.last_timestamp = None

        # For simulation: track the simulated time
        self.sim_start_real_time = datetime.now()
        self.sim_start_time = datetime.now()
        self.sim_time_factor = 1.0  # 1.0 = realtime, >1.0 = faster

    def _simulate_data(self):
        """Generate simulated energy data point."""
        # Current real and simulated time
        now_real = datetime.now()

        # Calculate simulated time
        elapsed_seconds = (now_real - self.sim_start_real_time).total_seconds()
        sim_elapsed = timedelta(seconds=elapsed_seconds * self.sim_time_factor)
        now_sim = self.sim_start_time + sim_elapsed

        # Extract temporal components
        hour = now_sim.hour
        weekday = now_sim.weekday()
        month = now_sim.month
        day_of_year = now_sim.timetuple().tm_yday

        # Configuration parameters
        base_demand = self.config['simulation']['base_demand']
        base_temp = self.config['simulation']['base_temperature']
        hour_factor = self.config['simulation']['hour_factor']
        weekday_factor = self.config['simulation']['weekday_factor']
        day_cycle_amplitude = self.config['simulation']['day_cycle_amplitude']
        season_cycle_amplitude = self.config['simulation']['season_cycle_amplitude']
        noise_factor = self.config['simulation']['noise_factor']
        temperature_noise = self.config['simulation']['temperature_noise']

        # Calculate temperature with seasonal and daily patterns
        # Seasonal component: highest in summer, lowest in winter
        # 172 is approx. day for June 21 (summer)
        season_factor = np.sin(2 * np.pi * (day_of_year - 172) / 365)
        seasonal_temp = season_cycle_amplitude * season_factor

        # Daily component: highest in afternoon, lowest pre-dawn
        daily_temp_factor = np.sin(2 * np.pi * (hour - 3) / 24)  # Peak at 3pm
        daily_temp = day_cycle_amplitude * daily_temp_factor

        # Combine temperature components
        temperature = base_temp + seasonal_temp + \
            daily_temp + np.random.normal(0, temperature_noise)

        # Calculate demand based on temporal patterns and temperature
        # Base demand
        demand = base_demand

        # Hourly pattern - higher in morning and evening
        hour_pattern = np.sin(np.pi * hour / 12) + 0.5 * \
            np.sin(np.pi * hour / 6)
        demand += hour_factor * hour_pattern

        # Weekday factor - higher on weekdays
        demand += weekday_factor if weekday < 5 else 0

        # Temperature impact - U-shaped relationship (high demand at both cold and hot temps)
        temp_factor = 0.1 * (temperature - 20)**2  # Minimum at 20°C
        demand += temp_factor

        # Add random noise
        demand += np.random.normal(0, noise_factor)

        # Ensure non-negative
        demand = max(0, demand)

        return {
            'timestamp': now_sim,
            'demand': demand,
            'temperature': temperature,
            'hour': hour,
            'day_of_week': weekday,
            'month': month,
            'year': now_sim.year
        }

    def _fetch_api_data(self):
        """Fetch energy data from API."""
        api_config = self.config['api']
        url = api_config['url']
        api_key = api_config['api_key']

        if not url:
            logger.error("API URL not configured")
            return None

        try:
            headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Process API response into our format
                timestamp = datetime.now()
                if 'timestamp' in data:
                    timestamp = pd.to_datetime(data['timestamp'])

                return {
                    'timestamp': timestamp,
                    'demand': data.get('demand') or data.get('consumption'),
                    'temperature': data.get('temperature'),
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.weekday(),
                    'month': timestamp.month,
                    'year': timestamp.year
                }
            else:
                logger.error(
                    f"API request failed with status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching API data: {e}")
            return None

    def _read_csv_data(self):
        """Read data from CSV file."""
        csv_config = self.config['csv']
        file_path = csv_config['path']

        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            return None

        try:
            # Read CSV into DataFrame if not already loaded
            if self.csv_data is None:
                self.csv_data = pd.read_csv(file_path)
                logger.info(
                    f"Loaded {len(self.csv_data)} records from {file_path}")

                # Ensure timestamp column
                if 'timestamp' in self.csv_data.columns:
                    self.csv_data['timestamp'] = pd.to_datetime(
                        self.csv_data['timestamp'])
                elif 'ds' in self.csv_data.columns:
                    self.csv_data['timestamp'] = pd.to_datetime(
                        self.csv_data['ds'])
                    if 'y' in self.csv_data.columns and 'demand' not in self.csv_data.columns:
                        self.csv_data['demand'] = self.csv_data['y']

                # Extract temporal features if needed
                if 'hour' not in self.csv_data.columns:
                    self.csv_data['hour'] = self.csv_data['timestamp'].dt.hour
                if 'day_of_week' not in self.csv_data.columns:
                    self.csv_data['day_of_week'] = self.csv_data['timestamp'].dt.dayofweek
                if 'month' not in self.csv_data.columns:
                    self.csv_data['month'] = self.csv_data['timestamp'].dt.month
                if 'year' not in self.csv_data.columns:
                    self.csv_data['year'] = self.csv_data['timestamp'].dt.year

            # Return next row
            if self.csv_index >= len(self.csv_data):
                if csv_config['loop']:
                    # Loop back to start
                    self.csv_index = 0
                else:
                    # End of data
                    logger.info("Reached end of CSV data")
                    return None

            # Get current row as dictionary
            row_dict = self.csv_data.iloc[self.csv_index].to_dict()
            self.csv_index += 1

            # Ensure timestamp is datetime
            if 'timestamp' in row_dict and not isinstance(row_dict['timestamp'], datetime):
                row_dict['timestamp'] = pd.to_datetime(row_dict['timestamp'])

            # Simulate real-time by replacing timestamp with current time
            if self.last_timestamp is None:
                # First reading, use current time
                self.last_timestamp = datetime.now()
            else:
                # Calculate time difference between rows in original data
                if self.csv_index > 1:
                    orig_prev = self.csv_data.iloc[self.csv_index -
                                                   2]['timestamp']
                    orig_curr = self.csv_data.iloc[self.csv_index -
                                                   1]['timestamp']
                    if isinstance(orig_prev, datetime) and isinstance(orig_curr, datetime):
                        time_diff = orig_curr - orig_prev
                        # Add this difference to last timestamp
                        self.last_timestamp += time_diff

            # Replace timestamp with simulated real-time
            row_dict['timestamp'] = self.last_timestamp

            return row_dict
        except Exception as e:
            logger.error(f"Error reading CSV data: {e}")
            return None

    def get_next_data_point(self):
        """Get next data point from the configured source."""
        source_type = self.config['source_type']

        if source_type == 'simulation':
            return self._simulate_data()
        elif source_type == 'api':
            return self._fetch_api_data()
        elif source_type == 'csv':
            return self._read_csv_data()
        else:
            logger.error(f"Unknown source type: {source_type}")
            return None

    def start(self, callback=None):
        """Start data source with callback for new data points."""
        if self.running:
            logger.warning("Data source already running")
            return

        self.callback = callback
        self.running = True
        self._reset_simulation_state()

        # Get interval from config based on source type
        source_type = self.config['source_type']
        interval = self.config[source_type]['interval_seconds']

        def data_loop():
            logger.info(
                f"Starting {source_type} data source, interval={interval}s")

            while self.running:
                try:
                    # Get next data point
                    data_point = self.get_next_data_point()

                    # Call callback if provided and data is valid
                    if data_point and self.callback:
                        self.callback(data_point)

                except Exception as e:
                    logger.error(f"Error in data source loop: {e}")

                # Wait for next interval
                time.sleep(interval)

            logger.info("Data source stopped")

        # Start data thread
        self.thread = threading.Thread(target=data_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the data source."""
        if not self.running:
            logger.warning("Data source not running")
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                logger.warning("Data source thread did not terminate cleanly")

        logger.info("Data source stopped")


# Example usage
if __name__ == "__main__":
    # Example callback function
    def data_callback(data_point):
        print(
            f"New data: {data_point['timestamp']} - Demand: {data_point['demand']:.2f}, Temp: {data_point['temperature']:.1f}°C")

    # Create and start data source
    source = EnergyDataSource()
    source.start(callback=data_callback)

    try:
        # Run for 30 seconds
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        source.stop()
