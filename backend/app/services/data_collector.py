import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.core.logging import loggers
from app.services.realtor_api import RealtorAPIService
from app.core.config import settings
from app.core.cache import property_cache, market_cache
from pathlib import Path

logger = loggers['ml']

class DataCollector:
    # Class-level cache so that **all** DataCollector instances (FastAPI dependency injection
    # creates many) see the same data. Keys are str(property_id).
    _shared_property_cache: dict[str, dict[str, Any]] = {}

    def __init__(self):
        """Initialize data collector with required services and paths to data files."""
        self.realtor_api = RealtorAPIService()
        # Local import to avoid circular dependencies
        from app.services.service_manager import ServiceManager
        # Reuse the singleton MarketDataService to avoid repeated heavy CSV loads
        self.market_service = ServiceManager.get_market_data_service()
        # Bind instance attribute to the shared cache defined at the class level.  This
        # ensures the cache survives across "per-request" DataCollector objects created
        # by FastAPI Depends(), eliminating the 404s we keep seeing when a new instance
        # doesn't have the search results cached.
        self.property_cache = DataCollector._shared_property_cache
        self.required_features = settings.REQUIRED_FEATURES
        self.data_dir = Path(settings.DATA_DIR)
        self.zillow_metro = pd.read_csv(self.data_dir / "zillow_metro.csv")
        self.zillow_county = pd.read_csv(self.data_dir / "zillow_county.csv")
        self.zillow_zip = pd.read_csv(self.data_dir / "zillow_zip.csv")
        self.realtor_metro = pd.read_csv(self.data_dir / "realtor_metro.csv")
        self.realtor_county = pd.read_csv(self.data_dir / "realtor_county.csv")
        self.realtor_zip = pd.read_csv(self.data_dir / "realtor_zip.csv", engine='python', na_values=' ')

    async def collect_training_data(self, location: str, time_period: str = "1y") -> pd.DataFrame:
        """
        Collect and prepare training data from multiple sources.
        
        Args:
            location: City, state or ZIP code
            time_period: Time period for historical data (e.g., "1y", "6m")
            
        Returns:
            DataFrame containing prepared training data
        """
        try:
            logger.info(f"Collecting training data for {location} over {time_period}")
            
            # Collect property data
            properties = await self._collect_property_data(location, time_period)
            if properties.empty:
                raise ValueError(f"No property data found for {location}")
            
            # Collect market data
            market_data = await self._collect_market_data(location, time_period)
            if market_data.empty:
                raise ValueError(f"No market data found for {location}")
            
            # Prepare training dataset
            training_data = self._prepare_training_dataset(properties, market_data)
            
            # Validate dataset
            self._validate_dataset(training_data)
            
            logger.info(f"Successfully collected {len(training_data)} training samples")
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {str(e)}")
            raise

    async def _collect_property_data(self, location: str, time_period: str) -> pd.DataFrame:
        """Collect historical property data from Realtor API."""
        try:
            # Get historical properties
            properties = await self.realtor_api.get_historical_properties(
                location=location,
                time_period=time_period
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(properties)
            
            # Clean and preprocess
            df = self._clean_property_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting property data: {str(e)}")
            raise

    async def _collect_market_data(self, location: str, time_period: str) -> pd.DataFrame:
        """Collect historical market data."""
        try:
            # Get market metrics
            if ',' in location:
                city, state = location.split(',')
                market_data = await self.market_service.get_metro_level_metrics(
                    city.strip(),
                    state.strip()
                )
            else:
                market_data = await self.market_service.get_zip_level_metrics(location.strip())
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            
            # Clean and preprocess
            df = self._clean_market_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting market data: {str(e)}")
            raise

    def _clean_property_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess property data."""
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            
            # Remove outliers
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[
                    (df[col] >= Q1 - 1.5 * IQR) &
                    (df[col] <= Q3 + 1.5 * IQR)
                ]
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning property data: {str(e)}")
            raise

    def _clean_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess market data."""
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
            
            # Calculate moving averages for smoothing
            for col in numeric_columns:
                df[f'{col}_ma'] = df[col].rolling(window=3).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning market data: {str(e)}")
            raise

    def _prepare_training_dataset(self, properties: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare final training dataset by combining and engineering features."""
        try:
            # Merge property and market data
            df = pd.merge(
                properties,
                market_data,
                on='date',
                how='inner'
            )
            
            # Calculate derived features
            df['price_per_sqft'] = df['price'] / df['sqft']
            df['beds_baths_ratio'] = df['beds'] / df['baths']
            df['property_age'] = datetime.now().year - df['year_built']
            df['sqft_per_bed'] = df['sqft'] / df['beds']
            
            # Calculate market-based features
            df['price_to_market_ratio'] = df['price'] / df['median_list_price']
            df['sqft_to_market_ratio'] = df['price_per_sqft'] / df['median_price_per_sqft']
            
            # Calculate comprehensive ROI
            # 1. Income
            annual_rent = df['estimated_rent'] * 12
            
            # 2. Expenses
            property_tax = df['price'] * 0.015  # 1.5% property tax
            insurance = df['price'] * 0.005     # 0.5% insurance
            maintenance = df['price'] * 0.01    # 1% maintenance
            property_management = annual_rent * 0.08  # 8% property management
            vacancy_loss = annual_rent * 0.05   # 5% vacancy rate
            
            # 3. Financing
            down_payment = df['price'] * 0.2    # 20% down payment
            loan_amount = df['price'] * 0.8     # 80% loan
            interest_rate = 0.06                # 6% interest rate
            loan_term = 30                      # 30-year loan
            monthly_payment = loan_amount * (interest_rate/12) * (1 + interest_rate/12)**(loan_term*12) / ((1 + interest_rate/12)**(loan_term*12) - 1)
            annual_mortgage = monthly_payment * 12
            
            # 4. Appreciation
            annual_appreciation = df['price'] * 0.03  # 3% annual appreciation
            
            # 5. Calculate ROI components
            annual_income = annual_rent - vacancy_loss
            annual_expenses = property_tax + insurance + maintenance + property_management + annual_mortgage
            annual_cash_flow = annual_income - annual_expenses
            annual_return = annual_cash_flow + annual_appreciation
            
            # 6. Calculate ROI
            df['roi'] = (annual_return / down_payment) * 100
            
            # Select and order features
            df = df[self.required_features + ['price', 'roi']]
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing training dataset: {str(e)}")
            raise

    def _validate_dataset(self, df: pd.DataFrame):
        """Validate the prepared dataset."""
        try:
            # Check for required features
            missing_features = set(self.required_features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                raise ValueError(f"Dataset contains missing values: {missing_values[missing_values > 0]}")
            
            # Check for infinite values
            infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum()
            if infinite_values.any():
                raise ValueError(f"Dataset contains infinite values: {infinite_values[infinite_values > 0]}")
            
            # Check for minimum data points
            if len(df) < 100:
                raise ValueError(f"Insufficient data points: {len(df)} < 100")
            
            logger.info("Dataset validation successful")
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            raise

    async def get_properties_by_location(
        self,
        city: Optional[str] = None,
        zip_code: Optional[str] = None,
        state_code: Optional[str] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        beds: Optional[int] = None,
        baths: Optional[int] = None,
        property_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get properties matching the search criteria from the Realtor API.
        
        Args:
            city: City name
            zip_code: ZIP code
            state_code: State code
            min_price: Minimum price
            max_price: Maximum price
            beds: Number of bedrooms
            baths: Number of bathrooms
            property_type: Type of property
            
        Returns:
            List of property dictionaries matching the criteria
        """
        try:
            # Use Realtor API to get live property data
            properties = await self.realtor_api.search_properties(
                city=city,
                state_code=state_code,
                zip_code=zip_code,
                min_price=min_price,
                max_price=max_price,
                beds=beds,
                baths=baths,
                property_type=property_type
            )
            
            if not properties:
                logger.warning(f"No properties found for location: {city}, {state_code} with the given criteria")
                return []
            
            # Format and return the properties
            formatted_properties = []
            for prop in properties:
                try:
                    formatted_prop = {
                        'property_id': prop.get('property_id', ''),
                        'address': prop.get('address', ''),
                        'city': prop.get('city', city),
                        'state': prop.get('state_code', state_code),
                        'zip_code': prop.get('zip_code', zip_code),
                        'price': float(prop.get('price', 0)),
                        'beds': int(prop.get('beds', 0)),
                        'baths': float(prop.get('baths', 0)),
                        'square_feet': float(prop.get('area', 0)),
                        'property_type': prop.get('property_type', 'single_family'),
                        'listing_date': prop.get('listing_date', ''),
                        'latitude': float(prop.get('latitude', 0)),
                        'longitude': float(prop.get('longitude', 0)),
                        'photo': prop.get('photo'),
                        'photos': prop.get('photos', []),
                        'link': prop.get('link'),
                        'days_on_market': prop.get('days_on_market')
                    }
                    # Cache the property for quick detail look-ups later
                    if formatted_prop.get('property_id') is not None:
                        # Cache in both old and new caches
                        prop_id = str(formatted_prop['property_id'])
                        cache_key = f"prop:{prop_id}"
                        property_cache.set(cache_key, formatted_prop)  # New cache with TTL
                        self.property_cache[prop_id] = formatted_prop  # Legacy cache
                    formatted_properties.append(formatted_prop)
                except Exception as e:
                    logger.error(f"Error formatting property: {str(e)}")
                    continue
            
            logger.info(f"Found {len(formatted_properties)} properties matching the criteria")
            return formatted_properties
            
        except Exception as e:
            logger.error(f"Error getting properties: {str(e)}")
            raise

    async def get_property_by_id(self, property_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific property by its ID.
        
        Args:
            property_id: Property identifier (can be numeric ID or MLS-style ID)
            
        Returns:
            Property dictionary if found, None otherwise
        """
        try:
            # Return from new cache first if available
            cache_key = f"prop:{str(property_id)}"
            if cached_property := property_cache.get(cache_key):
                logger.info(f"Property {property_id} found in cache")
                return cached_property

            # Fallback to old cache during transition
            if str(property_id) in self.property_cache:
                logger.info(f"Property {property_id} found in legacy cache")
                data = self.property_cache[str(property_id)]
                # Migrate to new cache
                property_cache.set(cache_key, data)
                return data

            logger.info(f"Getting property details for ID: {property_id}")
            
            # get_property_details is now an async method, so we await it
            property_data = await self.realtor_api.get_property_details(property_id)
            
            # Check if property is unavailable
            if not property_data or property_data.get("status") == "unavailable":
                error_msg = property_data.get('error', 'Unknown error') if property_data else 'No data returned'
                logger.warning(f"Property {property_id} is unavailable: {error_msg}")
                
                # Fallback: try search_properties with property_id filter
                try:
                    logger.info(f"Falling back to search_properties for property_id {property_id}")
                    props = await self.realtor_api.search_properties(property_id=property_id, limit=1)
                    if props and len(props) > 0:
                        # Cache the successful fallback result in both caches
                        property_cache.set(cache_key, props[0])
                        self.property_cache[str(property_id)] = props[0]  # Keep legacy cache in sync
                        return props[0]  # Already formatted in search_properties
                    else:
                        logger.warning(f"Fallback search_properties returned no results for {property_id}")
                except Exception as e:
                    logger.error(f"Fallback search_properties failed for {property_id}: {e}")
                return None

            # Validate property data
            if not isinstance(property_data, dict):
                logger.warning(f"Invalid property data format for {property_id}: expected dict, got {type(property_data)}")
                return None

            # If property_data already contains flattened keys (address, city, etc.) assume it's ready
            if property_data.get("address") and not property_data.get("location"):
                logger.info("Property data already formatted, caching and returning directly.")
                # Cache the formatted property in both caches
                property_cache.set(cache_key, property_data)
                self.property_cache[str(property_id)] = property_data  # Keep legacy cache in sync
                return property_data

            # Otherwise, format nested structure
            formatted_property = self._format_property(property_data)
            if formatted_property:
                logger.info(f"Successfully retrieved and formatted property {property_id}")
                # Cache the formatted property in both caches
                property_cache.set(cache_key, formatted_property)
                self.property_cache[str(property_id)] = formatted_property  # Keep legacy cache in sync
            else:
                logger.warning(f"Failed to format property data for {property_id}")
            return formatted_property

        except Exception as e:
            logger.error(f"Error getting property by ID {property_id}: {str(e)}")
            return None

    def _format_properties(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format DataFrame rows into property dictionaries."""
        properties = []
        for _, row in data.iterrows():
            property_dict = self._format_property(row)
            if property_dict:
                properties.append(property_dict)
        return properties

    def _format_property(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format a single property dictionary from the Realtor API response."""
        try:
            # Extract location data
            location = row.get("location", {})
            address = location.get("address", {})
            description = row.get("description", {})
            
            # Get photos
            photos = []
            if row.get("primary_photo"):
                photos.append(row["primary_photo"]["href"])
            if row.get("photos"):
                photos.extend([photo["href"] for photo in row["photos"]])
            
            # Format the property data
            property_dict = {
                'property_id': str(row.get('property_id', '')),
                'address': str(address.get('line', '')),
                'city': str(address.get('city', '')),
                'state': str(address.get('state_code', '')),
                'zip_code': str(address.get('postal_code', '')),
                'price': float(row.get('list_price', 0)),
                'beds': int(description.get('beds', 0)),
                'baths': float(description.get('baths', 0)),
                'square_feet': float(description.get('sqft', 0)),
                'property_type': str(description.get('type', 'single_family')),
                'listing_date': str(row.get('list_date', '')),
                'latitude': float(address.get('coordinate', {}).get('lat', 0)),
                'longitude': float(address.get('coordinate', {}).get('lon', 0)),
                'photo': row.get('primary_photo', {}).get('href') if row.get('primary_photo') else None,
                'photos': photos,
                'link': row.get('href'),
                'days_on_market': row.get('days_on_market'),
                'year_built': int(description.get('year_built', 0)),
                'lot_size': float(description.get('lot_sqft', 0)),
                'description': str(description.get('text', '')),
                'features': list(description.get('features', []))
            }
            
            # Validate required fields
            required_fields = ['property_id', 'address', 'city', 'state', 'price']
            missing_fields = [field for field in required_fields if not property_dict.get(field)]
            if missing_fields:
                logger.warning(f"Missing required fields for property {property_dict.get('property_id')}: {missing_fields}")
                return None
                
            return property_dict
            
        except Exception as e:
            logger.error(f"Error formatting property: {str(e)}")
            return None 