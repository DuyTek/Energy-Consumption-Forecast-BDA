# models/prophet_model.py
import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import os
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnergyProphetModel")


class EnergyProphetModel:
    """Prophet model for energy consumption forecasting."""

    def __init__(self, model_path="models/saved/prophet_energy_model.pkl"):
        """Initialize the Prophet model.

        Args:
            model_path: Path to save/load the model
        """
        self.model = None
        self.model_path = model_path
        self.last_retraining = None

        # Create directory for models if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Try to load an existing model
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded existing model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model = None

    # Modified _prepare_data method for prophet_model.py

    def _prepare_data(self, df):
        """Prepare dataframe for Prophet model."""
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()

        # Prophet requires columns 'ds' and 'y'
        if 'ds' not in df.columns and 'timestamp' in df.columns:
            df['ds'] = df['timestamp']

        if 'y' not in df.columns and 'demand' in df.columns:
            df['y'] = df['demand']

        # Ensure ds is datetime type
        df['ds'] = pd.to_datetime(df['ds'])

        # Create the base prophet dataframe
        prophet_df = df[['ds', 'y']]

        # Add temperature as regressor if available, with NaN handling
        if 'temperature' in df.columns:
            # Check for NaN values in temperature
            if df['temperature'].isna().any():
                logger.warning(
                    f"Found {df['temperature'].isna().sum()} NaN values in temperature column")

                # Option 1: Fill NaN with median temperature (safer than mean for outliers)
                temp_median = df['temperature'].median()
                prophet_df['temperature'] = df['temperature'].fillna(
                    temp_median)
                logger.info(
                    f"Filled NaN temperature values with median: {temp_median}")

                # Option 2 (alternative): Drop rows with NaN temperature
                # prophet_df = prophet_df[~df['temperature'].isna()]
                # logger.info(f"Dropped {len(df) - len(prophet_df)} rows with NaN temperature values")
            else:
                # No NaN values, just copy the column
                prophet_df['temperature'] = df['temperature']

        return prophet_df

    # Modified train method to properly handle temperature inclusion

    def train(self, df, include_temperature=None):
        """Train the Prophet model.

        Args:
            df: DataFrame with training data
            include_temperature: Whether to include temperature as a regressor.
                                If None, will be determined by presence of temperature column.
        """
        # Prepare data
        prophet_df = self._prepare_data(df)

        # Determine whether to include temperature
        has_temperature = 'temperature' in prophet_df.columns
        if include_temperature is None:
            # Auto-detect based on data presence
            include_temperature = has_temperature

        # Validate temperature usage
        if include_temperature and not has_temperature:
            logger.warning(
                "Temperature inclusion requested but no temperature data available")
            include_temperature = False

        # Initialize a new Prophet model
        model = Prophet()

        # Add temperature as a regressor if requested and available
        if include_temperature:
            logger.info("Adding temperature as a regressor in Prophet model")
            model.add_regressor('temperature')

        # Fit the model
        logger.info("Training Prophet model...")
        model.fit(prophet_df)
        logger.info("Model training completed")

        self.model = model
        self.last_retraining = datetime.now()

        # Save the model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {self.model_path}")

        return self.model

    def generate_temp_forecast(self, historical_temps, periods):
        """Generate simple temperature forecast based on historical patterns."""
        # Handle empty or too short historical data
        if historical_temps.empty:
            # If no historical temperature data, use a reasonable default value
            return np.ones(periods) * 20.0  # Default 20°C

        # Convert to numpy array if it's a Series
        if hasattr(historical_temps, 'values'):
            historical_temps = historical_temps.values

        # Option 1: Use last year's temperatures if available
        if len(historical_temps) >= periods:
            # Return the last 'periods' values
            return historical_temps[-periods:]

        # Option 2: Repeat the available data if not enough historical data
        # This ensures we always return an array of the correct length
        repetitions = (periods // len(historical_temps)) + 1
        repeated_temps = np.tile(historical_temps, repetitions)
        return repeated_temps[:periods]  # Slice to exact number of periods

    def predict(self, periods=365, include_history=True, temperature_forecast=None):
        """Generate forecasts for future periods."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)

        # Check if temperature is used as a regressor
        has_temp_regressor = False
        for name, _ in self.model.extra_regressors.items():
            if name == 'temperature':
                has_temp_regressor = True
                break

        # Add temperature data for future periods
        if has_temp_regressor:
            if temperature_forecast is None:
                # If no temperature forecast provided, generate a simple one
                if 'temperature' in self.model.history.columns:
                    historical_temps = self.model.history['temperature']
                    # Create synthetic temperature forecast based on historical patterns
                    temp_forecast = self.generate_temp_forecast(
                        historical_temps, periods)
                else:
                    # If no temperature in history (unlikely but possible)
                    # Default temperature
                    temp_forecast = np.ones(periods) * 20.0
            else:
                temp_forecast = temperature_forecast

            # Ensure temp_forecast has the correct length
            if len(temp_forecast) != periods:
                raise ValueError(f"Temperature forecast length ({len(temp_forecast)}) "
                                 f"must match forecast periods ({periods})")

            # Add to future dataframe
            # Make sure temperature column exists
            if 'temperature' not in future.columns:
                future['temperature'] = np.nan

            # Add forecast values to future periods
            future_rows = len(future) - periods
            future.loc[future_rows:, 'temperature'] = temp_forecast

        # Generate forecasts
        forecast = self.model.predict(future)
        return forecast

    def plot_forecast(self, forecast=None, periods=365):
        """Plot the forecast."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if forecast is None:
            forecast = self.predict(periods=periods)

        # Plot forecast
        fig = self.model.plot(forecast)
        return fig

    def plot_components(self, forecast=None, periods=365):
        """Plot the components of the forecast."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if forecast is None:
            forecast = self.predict(periods=periods)

        # Plot components
        fig = self.model.plot_components(forecast)
        return fig

    def real_time_update(self, new_data_point):
        """Update model with new real-time data point.

        Note: Prophet doesn't support true online learning,
        so this method will collect data for eventual retraining.
        """
        # In a real implementation, you might:
        # 1. Append the new data to a buffer or database
        # 2. Check if enough new data is collected to trigger retraining
        # 3. Retrain the model periodically

        # For now, just log the update
        logger.info(f"Received real-time data point: {new_data_point}")
        return True
