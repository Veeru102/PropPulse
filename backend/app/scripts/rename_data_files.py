import os
from pathlib import Path
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File mapping
FILE_MAPPING = {
    'RDC_Inventory_Core_Metrics_Metro.csv': 'realtor_metro.csv',
    'RDC_Inventory_Core_Metrics_County.csv': 'realtor_county.csv',
    'RDC_Inventory_Core_Metrics_Zip.csv': 'realtor_zip.csv',
    'RDC_Inventory_Core_Metrics_Metro_History.csv': 'realtor_metro_history.csv',
    'RDC_Inventory_Core_Metrics_County_History.csv': 'realtor_county_history.csv',
    'RDC_Inventory_Core_Metrics_Zip_History.csv': 'realtor_zip_history.csv'
}

def rename_files():
    """Rename downloaded files to match expected names."""
    try:
        # Get the data directory
        data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        
        # Rename files
        for old_name, new_name in FILE_MAPPING.items():
            old_path = data_dir / old_name
            new_path = data_dir / new_name
            
            if old_path.exists():
                # If new file already exists, remove it
                if new_path.exists():
                    new_path.unlink()
                
                # Rename the file
                shutil.move(str(old_path), str(new_path))
                logger.info(f"Renamed {old_name} to {new_name}")
            else:
                logger.warning(f"File {old_name} not found")
        
        logger.info("File renaming completed")
        
    except Exception as e:
        logger.error(f"Error renaming files: {e}")
        raise

if __name__ == "__main__":
    rename_files() 