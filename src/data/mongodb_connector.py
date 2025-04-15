# data/mongodb_connector.py
import pymongo
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import logging
import os
import json
import urllib
import certifi
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MongoDBConnector")


class EnergyDatabaseConnector:
    """Connector for MongoDB operations with energy consumption data."""
    username = os.environ.get("MONGODB_USER")
    password = os.environ.get("MONGODB_PASSWORD")

    encoded_username = urllib.parse.quote_plus(username) if username else None
    encoded_password = urllib.parse.quote_plus(password) if password else None

    def __init__(self, connection_string=None, db_name="energy_consumption_bda"):
        """Initialize database connection."""
        # Use environment variable if connection string not provided
        if connection_string is None and self.encoded_username and self.encoded_password:
            connection_string = f"mongodb+srv://{self.encoded_username}:{self.encoded_password}@energy-consumption-bda.obgdd.mongodb.net/?retryWrites=true&w=majority&appName=energy-consumption-bda",

        self.client = None
        self.db = None
        self.consumption_collection = None

        logger.info(
            f"Connecting to MongoDB with connection string: {connection_string}")
        try:
            self.client = MongoClient(
                connection_string, tlsCAFile=certifi.where())
            self.db = self.client[db_name]

            # Create collections with time-series optimization
            self.consumption_collection = self.db["consumption_data"]

            # Create indexes for faster querying
            self.consumption_collection.create_index(
                [("timestamp", pymongo.ASCENDING)])
            self.consumption_collection.create_index(
                [("hour", pymongo.ASCENDING)])
            self.consumption_collection.create_index(
                [("day_of_week", pymongo.ASCENDING)])
            self.consumption_collection.create_index(
                [("month", pymongo.ASCENDING)])

            logger.info(f"Connected to MongoDB: {db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            # Initialize a fallback CSV storage mechanism
            self._init_csv_fallback()

    def _init_csv_fallback(self):
        """Initialize CSV fallback storage when MongoDB is unavailable."""
        logger.warning("Initializing CSV fallback storage for data")
        self.csv_fallback_dir = "data/csv_fallback"
        os.makedirs(self.csv_fallback_dir, exist_ok=True)

        # Create empty DataFrame if no existing data
        self.csv_fallback_file = f"{self.csv_fallback_dir}/energy_data.csv"
        if not os.path.exists(self.csv_fallback_file):
            empty_df = pd.DataFrame(columns=['timestamp', 'demand', 'temperature',
                                             'hour', 'day_of_week', 'month', 'year'])
            empty_df.to_csv(self.csv_fallback_file, index=False)

    def is_connected(self):
        """Check if connected to MongoDB."""
        return self.client is not None

    def store_dataframe(self, df, collection_name="consumption_data"):
        """Store pandas DataFrame in the database."""
        if df is None or len(df) == 0:
            logger.warning("Empty DataFrame provided, nothing to store")
            return

        # Make sure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'ds' in df.columns:
            # If using Prophet format, convert ds to timestamp
            df['timestamp'] = pd.to_datetime(df['ds'])

        # Extract temporal features if not present
        if 'timestamp' in df.columns:
            if 'hour' not in df.columns:
                df['hour'] = df['timestamp'].dt.hour
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['timestamp'].dt.dayofweek
            if 'month' not in df.columns:
                df['month'] = df['timestamp'].dt.month
            if 'year' not in df.columns:
                df['year'] = df['timestamp'].dt.year

        # Convert DataFrame to list of dictionaries
        records = df.to_dict("records")

        if self.is_connected():
            try:
                # Insert into MongoDB collection
                self.db[collection_name].insert_many(records)
                logger.info(
                    f"Inserted {len(records)} records into {collection_name}")
            except Exception as e:
                logger.error(f"Error inserting data into MongoDB: {e}")
                self._store_csv_fallback(df)
        else:
            # Use CSV fallback
            self._store_csv_fallback(df)

    def _store_csv_fallback(self, df):
        """Store data in CSV file when MongoDB is unavailable."""
        try:
            # If file exists, append without writing headers
            if os.path.exists(self.csv_fallback_file):
                # Read existing data to avoid duplicates
                existing_df = pd.read_csv(self.csv_fallback_file)
                combined_df = pd.concat([existing_df, df]).drop_duplicates(
                    subset=['timestamp'])
                combined_df.to_csv(self.csv_fallback_file, index=False)
            else:
                df.to_csv(self.csv_fallback_file, index=False)

            logger.info(f"Stored {len(df)} records in CSV fallback")
        except Exception as e:
            logger.error(f"Error storing data in CSV fallback: {e}")

    def fetch_data(self, start_date=None, end_date=None, collection_name="consumption_data"):
        """Fetch data between date range."""
        if self.is_connected():
            try:
                query = {}
                if start_date and end_date:
                    query["timestamp"] = {"$gte": start_date, "$lte": end_date}

                cursor = self.db[collection_name].find(query)
                result_df = pd.DataFrame(list(cursor))

                # Remove MongoDB ID field
                if '_id' in result_df.columns:
                    result_df = result_df.drop('_id', axis=1)

                logger.info(f"Fetched {len(result_df)} records from MongoDB")
                return result_df
            except Exception as e:
                logger.error(f"Error fetching data from MongoDB: {e}")
                return self._fetch_csv_fallback(start_date, end_date)
        else:
            # Use CSV fallback
            return self._fetch_csv_fallback(start_date, end_date)

    def _fetch_csv_fallback(self, start_date=None, end_date=None):
        """Fetch data from CSV file when MongoDB is unavailable."""
        try:
            if not os.path.exists(self.csv_fallback_file):
                logger.warning("CSV fallback file does not exist")
                return pd.DataFrame()

            df = pd.read_csv(self.csv_fallback_file)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter by date range if provided
            if start_date and end_date:
                df = df[(df['timestamp'] >= start_date)
                        & (df['timestamp'] <= end_date)]

            logger.info(f"Fetched {len(df)} records from CSV fallback")
            return df
        except Exception as e:
            logger.error(f"Error fetching data from CSV fallback: {e}")
            return pd.DataFrame()

    def store_real_time_data(self, data_point, collection_name="consumption_data"):
        """Store a single real-time data point."""
        # Ensure timestamp is present
        if "timestamp" not in data_point:
            data_point["timestamp"] = datetime.now()
        elif not isinstance(data_point["timestamp"], datetime):
            data_point["timestamp"] = pd.to_datetime(data_point["timestamp"])

        # Extract temporal features
        dt = data_point["timestamp"]
        data_point["hour"] = dt.hour
        data_point["day_of_week"] = dt.weekday()
        data_point["month"] = dt.month
        data_point["year"] = dt.year

        if self.is_connected():
            try:
                # Insert into MongoDB collection
                self.db[collection_name].insert_one(data_point)
                logger.info(
                    f"Inserted real-time data point into {collection_name}")
            except Exception as e:
                logger.error(
                    f"Error inserting real-time data into MongoDB: {e}")
                self._store_realtime_csv_fallback(data_point)
        else:
            # Use CSV fallback
            self._store_realtime_csv_fallback(data_point)

    def _store_realtime_csv_fallback(self, data_point):
        """Store real-time data point in CSV when MongoDB is unavailable."""
        df = pd.DataFrame([data_point])
        self._store_csv_fallback(df)

    def load_from_csv(self, csv_path):
        """Load data from CSV file into the database."""
        try:
            df = pd.read_csv(csv_path)

            # Ensure timestamp column is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'ds' in df.columns:
                # If using Prophet format (ds/y)
                df['timestamp'] = pd.to_datetime(df['ds'])
                if 'y' in df.columns and 'demand' not in df.columns:
                    df['demand'] = df['y']

            # Store in database
            self.store_dataframe(df)
            logger.info(f"Loaded {len(df)} records from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from CSV {csv_path}: {e}")
            return None

    def close_connection(self):
        """Close the database connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")
            self.client = None
