import os
import sys
import pandas as pd
import requests
from pathlib import Path
import logging

# Try to import BASE_DIR from config, else define it here
try:
    from app.core.config import settings, BASE_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    class DummySettings:
        DATA_DIR = "data"
        MODEL_DIR = "data/ml_models"
    settings = DummySettings()

logger = logging.getLogger(__name__)

# Note: Due to Realtor.com's access restrictions, these files must be downloaded manually
# from https://www.realtor.com/research/data/ and placed in the data directory.
# The script will process them after they are downloaded.

# URLs for Zillow data (these are still publicly accessible)
ZILLOW_URLS = {
    'zip': 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'county': 'https://files.zillowstatic.com/research/public_csvs/zhvi/County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'metro': 'https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
}

def ensure_data_directory():
    """Ensure the data directory exists."""
    data_dir = os.path.join(str(BASE_DIR), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data directory at {data_dir}")
    return data_dir

def download_file(url: str, output_path: str) -> bool:
    """Download a file from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded {url} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def process_realtor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process Realtor.com data to extract relevant columns."""
    try:
        # Select and rename relevant columns
        columns = {
            'postal_code': 'zip_code',
            'median_list_price': 'median_list_price',
            'median_dom': 'median_dom',
            'median_list_price_per_sqft': 'median_list_price_per_sqft',
            'city': 'city',
            'state_code': 'state_code',
            'price_change_1y': 'price_change_1y',
            'price_change_5y': 'price_change_5y',
            'inventory': 'inventory',
            'inventory_change_1y': 'inventory_change_1y',
            'inventory_change_5y': 'inventory_change_5y',
            'days_to_sell': 'days_to_sell',
            'days_to_sell_change_1y': 'days_to_sell_change_1y',
            'days_to_sell_change_5y': 'days_to_sell_change_5y',
            'price_to_rent_ratio': 'price_to_rent_ratio',
            'price_to_rent_ratio_change_1y': 'price_to_rent_ratio_change_1y',
            'price_to_rent_ratio_change_5y': 'price_to_rent_ratio_change_5y',
            'price_to_income_ratio': 'price_to_income_ratio',
            'price_to_income_ratio_change_1y': 'price_to_income_ratio_change_1y',
            'price_to_income_ratio_change_5y': 'price_to_income_ratio_change_5y'
        }
        
        # Filter columns that exist in the dataframe
        existing_columns = {k: v for k, v in columns.items() if k in df.columns}
        
        # Select and rename columns
        processed_df = df[existing_columns.keys()].rename(columns=existing_columns)
        
        # Calculate additional metrics if possible
        if 'median_list_price' in processed_df.columns and 'median_list_price_per_sqft' in processed_df.columns:
            processed_df['price_to_median_ratio'] = processed_df['median_list_price'] / processed_df['median_list_price'].median()
            processed_df['price_to_avg_ratio'] = processed_df['median_list_price'] / processed_df['median_list_price'].mean()
        
        return processed_df
    except Exception as e:
        logger.error(f"Error processing Realtor.com data: {e}")
        return df

def process_zillow_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process Zillow data to extract relevant columns."""
    try:
        # Select and rename relevant columns
        columns = {
            'RegionID': 'region_id',
            'RegionName': 'region_name',
            'State': 'state_code',
            'City': 'city',
            'Metro': 'metro',
            'CountyName': 'county'
        }
        
        # Filter columns that exist in the dataframe
        existing_columns = {k: v for k, v in columns.items() if k in df.columns}
        
        # Select and rename columns
        processed_df = df[existing_columns.keys()].rename(columns=existing_columns)
        
        # Extract the most recent price data
        price_columns = [col for col in df.columns if col.startswith('20')]
        if price_columns:
            latest_price = price_columns[-1]
            processed_df['median_list_price'] = df[latest_price]
        
        return processed_df
    except Exception as e:
        logger.error(f"Error processing Zillow data: {e}")
        return df

def main():
    """Main function to download and process market data."""
    data_dir = ensure_data_directory()
    
    # Process Realtor.com data (assuming files are already downloaded)
    realtor_files = [
        'realtor_metro.csv',
        'realtor_county.csv',
        'realtor_zip.csv',
        'realtor_metro_history.csv',
        'realtor_county_history.csv',
        'realtor_zip_history.csv'
    ]
    
    for filename in realtor_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                processed_df = process_realtor_data(df)
                processed_df.to_csv(file_path, index=False)
                logger.info(f"Processed {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
        else:
            logger.warning(f"File {filename} not found. Please download it from https://www.realtor.com/research/data/")
    
    # Download and process Zillow data
    for data_type, url in ZILLOW_URLS.items():
        output_path = os.path.join(data_dir, f'zillow_{data_type}.csv')
        if download_file(url, output_path):
            try:
                df = pd.read_csv(output_path)
                processed_df = process_zillow_data(df)
                processed_df.to_csv(output_path, index=False)
                logger.info(f"Processed Zillow {data_type} data")
            except Exception as e:
                logger.error(f"Error processing Zillow {data_type} data: {e}")

if __name__ == "__main__":
    main() 