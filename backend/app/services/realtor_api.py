import aiohttp
from typing import Optional, List, Dict, Any
from app.core.config import settings
import logging
import ssl
import certifi
import json
from datetime import date, datetime # Import date for calculating days on market and datetime for parsing ISO 8601 string
import requests
import time
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtorAPIService:
    def __init__(self):
        self.api_key = settings.REALTOR_API_KEY
        if not self.api_key:
            logger.error("RAPIDAPI_KEY is not set in environment variables")
            raise ValueError("RAPIDAPI_KEY is required but not set")
        
        self.base_url = "https://realty-in-us.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com",
            "Content-Type": "application/json"
        }
        # Create SSL context with verified certificates
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        logger.info(f"Initialized RealtorAPIService with base_url: {self.base_url}")
        logger.info(f"API Key length: {len(self.api_key)}")
        logger.info(f"API Key first 4 chars: {self.api_key[:4]}")

    async def search_properties(
        self,
        *,
        city: str | None = None,
        state_code: str | None = None,
        zip_code: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        min_price: int | None = None,
        max_price: int | None = None,
        beds: int | None = None,
        baths: int | None = None,
        property_type: str | None = None,
        property_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Search for properties using the Realtor.com API v3.
        """
        # Create a connector with the SSL context
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Use the properties/v3/list endpoint
            url = f"{self.base_url}/properties/v3/list"
            
            # Build the payload according to v3 specifications
            payload: dict[str, Any] = {
                "limit": limit,
                "offset": offset,
                "status": ["for_sale"],
                "sort": {
                    "field": "list_date",
                    "direction": "desc"
                }
            }

            # If property_id is provided, use it to search for a specific property
            if property_id:
                payload["property_id"] = property_id
            else:
                # Add location parameters (pick one)
                if zip_code:
                    payload["postal_code"] = zip_code
                elif city and state_code:
                    payload["city"] = city
                    payload["state_code"] = state_code
                elif latitude and longitude:
                    payload["lat"] = latitude
                    payload["lon"] = longitude
                else:
                    logger.error("No location parameters provided (city/state, zip_code, or lat/lon)")
                    return []

                # Add price parameters
                if min_price is not None:
                    payload["price_min"] = max(0, min_price)
                if max_price is not None:
                    payload["price_max"] = min(100000000, max_price)

                # Add other parameters
                if beds is not None:
                    payload["beds_min"] = max(0, beds)
                if baths is not None:
                    payload["baths_min"] = max(0, baths)
                if property_type:
                    payload["prop_type"] = [property_type]  # Must be an array
                else:
                    payload["prop_type"] = ["single_family"]  # Default property type

            # Log the request details
            logger.info("=== REALTOR API REQUEST ===")
            logger.info(f"URL: {url}")
            logger.debug(f"Headers: {json.dumps(self.headers, indent=2)}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

            try:
                async with session.post(url, headers=self.headers, json=payload) as resp:
                    logger.info("=== REALTOR API RESPONSE ===")
                    logger.info(f"Status: {resp.status}")
                    logger.debug(f"Headers: {json.dumps(dict(resp.headers), indent=2)}")
                    
                    response_text = await resp.text()
                    logger.debug(f"Raw Response: {response_text[:1000]} ... (truncated)")
                    
                    if resp.status == 200:
                        try:
                            data = json.loads(response_text)
                            logger.debug(f"Parsed Response: {json.dumps(data, indent=2)[:1500]} ... (truncated)")
                            
                            # Extract results from v3 response structure with proper null safety
                            if not data:
                                logger.warning("API returned empty/null data")
                                return []
                                
                            data_section = data.get("data")
                            if not data_section:
                                logger.warning("API response missing 'data' section")
                                return []
                                
                            home_search = data_section.get("home_search")
                            if not home_search:
                                logger.warning("API response missing 'home_search' section")
                                return []
                                
                            results = home_search.get("results")
                            if not results:
                                logger.warning("API response missing 'results' section or results is empty")
                                return []
                            
                            if results:
                                # Filter results based on criteria
                                filtered_results = []
                                for result in results:
                                    if not result or not isinstance(result, dict):
                                        logger.warning(f"Skipping invalid result: {result}")
                                        continue
                                        
                                    # Get property details with null safety
                                    price = result.get("list_price") # Removed default 0, will handle None explicitly
                                    description = result.get("description") or {}
                                    if not isinstance(description, dict):
                                        description = {}
                                    beds_count = description.get("beds")
                                    baths_count = description.get("baths")
                                    
                                    # Ensure price is not None before comparison
                                    if price is None:
                                        logger.debug(f"Skipping property {result.get('property_id')} due to null price.")
                                        continue

                                    # Check if property meets criteria
                                    if min_price is not None and price < min_price:
                                        continue
                                    if max_price is not None and price > max_price:
                                        continue
                                    if beds is not None and (beds_count is None or beds_count < beds):
                                        continue
                                    if baths is not None and (baths_count is None or baths_count < baths):
                                        continue
                                    
                                    filtered_results.append(result)
                                
                                formatted_properties = self._format_properties(filtered_results)
                                logger.info(f"Found {len(formatted_properties)} properties after filtering")
                                return formatted_properties
                            else:
                                logger.warning("No properties found in response")
                                return []
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON response: {e}")
                            return []
                    else:
                        logger.error(f"API request failed with status {resp.status}: {response_text}")
                        return []
                    
            except Exception as e:
                logger.error(f"Error during API request: {str(e)}")
                raise

    async def get_property_details(self, property_id: str) -> Dict[str, Any]:
        """
        Fetch detailed data for a property using the working v3/list endpoint.
        This replaces the previous approach which used broken v3/detail endpoints.
        
        Args:
            property_id: The property ID to fetch details for
            
        Returns:
            Formatted property dictionary or error dict if unavailable
        """
        if not property_id:
            logger.error("get_property_details called with empty property_id")
            return {"property_id": property_id, "error": "Invalid property ID", "status": "unavailable"}
        
        logger.info(f"Fetching property details for ID: {property_id}")
        
        # Use the working search_properties method with property_id filter
        # This uses the /properties/v3/list endpoint which is confirmed working
        try:
            # Try with retry logic for rate limiting
            for attempt in range(2):  # Allow one retry
                try:
                    # Use search_properties since it uses the working v3/list endpoint
                    properties = await self.search_properties(property_id=property_id, limit=1)
                    
                    if not properties:
                        logger.warning(f"No property found for ID: {property_id}")
                        return {"property_id": property_id, "error": "Property not found", "status": "unavailable"}
                    
                    property_data = properties[0]
                    
                    # Validate that we got valid property data
                    if not property_data or not isinstance(property_data, dict):
                        logger.warning(f"Invalid property data received for ID: {property_data}")
                        return {"property_id": property_id, "error": "Invalid property data", "status": "unavailable"}
                    
                    # The search_properties method already returns formatted data
                    # But we need to convert it to the format expected by _format_property_details
                    # Check if this is already in the expected format
                    if property_data.get("address") and not property_data.get("location"):
                        # Already formatted by search_properties, convert to expected format
                        formatted_details = {
                            "property_id": str(property_data.get("property_id", property_id)),
                            "address": str(property_data.get("address", "")),
                            "city": str(property_data.get("city", "")),
                            "state": str(property_data.get("state_code", "")),
                            "zip_code": str(property_data.get("zip_code", "")),
                            "price": property_data.get("price", 0),
                            "beds": property_data.get("beds", 0),
                            "baths": property_data.get("baths", 0),
                            "square_feet": property_data.get("area", 0),  # Note: search_properties uses 'area'
                            "property_type": str(property_data.get("property_type", "single_family")),
                            "listing_date": str(property_data.get("listing_date", "")),
                            "latitude": property_data.get("latitude", 0.0),
                            "longitude": property_data.get("longitude", 0.0),
                            "description": str(property_data.get("description", "")),
                            "features": property_data.get("features", []),
                            "photos": property_data.get("photos", []),
                            "year_built": property_data.get("year_built", 0),
                            "lot_size": property_data.get("lot_size", 0)
                        }
                        logger.info(f"Successfully retrieved property details for ID: {property_id}")
                        return formatted_details
                    else:
                        # Raw property data from API, use existing formatter
                        formatted_details = self._format_property_details(property_data)
                        logger.info(f"Successfully retrieved and formatted property details for ID: {property_id}")
                        return formatted_details
                        
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        if attempt == 0:  # First attempt failed due to rate limiting
                            retry_delay = 5  # Wait 5 seconds before retry
                            logger.warning(f"Rate limited fetching property {property_id}. Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            logger.error(f"Rate limited again for property {property_id}. Giving up.")
                            return {"property_id": property_id, "error": "Rate limited", "status": "unavailable"}
                    else:
                        # Non-rate-limit error, don't retry
                        logger.error(f"Error fetching property details for {property_id}: {str(e)}")
                        return {"property_id": property_id, "error": str(e), "status": "unavailable"}
                        
                # If we reach here, the attempt succeeded
                break
                
        except Exception as e:
            logger.error(f"Unexpected error fetching property details for {property_id}: {str(e)}")
            return {"property_id": property_id, "error": "Unexpected error occurred", "status": "unavailable"}

    def _format_properties(self, properties: List[Dict]) -> List[Dict]:
        formatted_list = []
        today = date.today() # Get today's date once for efficiency
        
        if not properties:
            logger.warning("_format_properties called with empty or None properties list")
            return []
        
        for prop in properties:
            try:
                # Validate property data
                if not prop or not isinstance(prop, dict):
                    logger.warning(f"Skipping invalid property data: {prop}")
                    continue
                    
                logger.debug(f"Raw property data before formatting: {json.dumps(prop, indent=2)}")
                
                # Extract location data with null safety
                location = prop.get("location") or {}
                if not isinstance(location, dict):
                    location = {}
                    
                address = location.get("address") or {}
                if not isinstance(address, dict):
                    address = {}
                
                # Get photos from the property data with null safety
                photos = []
                primary_photo = prop.get("primary_photo")
                if primary_photo and isinstance(primary_photo, dict) and primary_photo.get("href"):
                    photos.append(primary_photo["href"])
                
                prop_photos = prop.get("photos")
                if prop_photos and isinstance(prop_photos, list):
                    for photo in prop_photos:
                        if photo and isinstance(photo, dict) and photo.get("href"):
                            photos.append(photo["href"])
                
                # Calculate days_on_market with more robust error handling
                days_on_market = None
                list_date_str = prop.get("list_date")
                if list_date_str and isinstance(list_date_str, str):
                    try:
                        list_date_obj = datetime.fromisoformat(list_date_str[:10]).date()
                        days_on_market = (today - list_date_obj).days
                        # Ensure days_on_market is not negative
                        if days_on_market < 0:
                            days_on_market = 0
                    except ValueError as e:
                        logger.warning(f"Could not parse list_date '{list_date_str}' for property {prop.get('property_id')}: {e}")
                    except Exception as e:
                         logger.warning(f"Error calculating days_on_market for property {prop.get('property_id')}: {e}")

                # Extract description data with null safety
                description = prop.get("description") or {}
                if not isinstance(description, dict):
                    description = {}
                
                # Determine IDs
                numeric_id = prop.get("property_id")  # Usually 10-digit ID preferred for detail endpoint
                mls_href_id = None
                href = prop.get("href")
                if href and isinstance(href, str):
                    # Extract MLS-style ID from href like ..._M58534-35579
                    try:
                        mls_href_id = href.split("_")[-1]
                    except Exception:
                        pass

                # Choose property_id field to be numeric if available else mls
                property_id_final = numeric_id or mls_href_id
                if not property_id_final:
                    source = prop.get("source")
                    if source and isinstance(source, dict):
                        property_id_final = source.get("listing_id")

                if not property_id_final:
                    logger.warning(f"Could not determine property_id for property: {prop}")
                    continue
                
                # Extract coordinate data with null safety
                coordinate = address.get("coordinate") or {}
                if not isinstance(coordinate, dict):
                    coordinate = {}
                
                formatted_list.append({
                    "property_id": str(property_id_final),
                    "mls_id": mls_href_id,
                    "address": f"{address.get('line', '')}, {address.get('city', '')}, {address.get('state_code', '')} {address.get('postal_code', '')}",
                    "city": address.get("city"),
                    "state_code": address.get("state_code"),
                    "zip_code": address.get("postal_code"),
                    "price": prop.get("list_price"),
                    "beds": description.get("beds"),
                    "baths": description.get("baths"),
                    "area": description.get("sqft"),
                    "property_type": description.get("type", "single_family"),
                    "latitude": coordinate.get("lat"),
                    "longitude": coordinate.get("lon"),
                    "photo": primary_photo.get("href") if primary_photo and isinstance(primary_photo, dict) else None,
                    "photos": photos,
                    "link": prop.get("href"),
                    "days_on_market": days_on_market,
                    "listing_date": prop.get("list_date")
                })
            except Exception as e:
                logger.error(f"Error formatting property data (outside of date parsing): {e} for property: {prop}")
                continue
        return formatted_list

    async def enrich_listing(self, listing: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich a single listing with detailed data if missing.
        """
        # if either field is missing, hit the detail endpoint once
        # Check if days_on_market is None after initial formatting
        needs_dom = listing.get("days_on_market") is None or listing.get("days_on_market", 0) == 0
        needs_sqft = not listing.get("area") # Using 'area' key from formatted data
        
        if not (needs_dom or needs_sqft):
            return listing

        # Create a connector with the SSL context if not already done (or use instance session)
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            url = f"{self.base_url}/properties/v3/detail"
            property_id = listing.get("property_id")
            
            if not property_id:
                logger.warning("Cannot enrich listing: property_id is missing.")
                return listing
                
            logger.info(f"Attempting to enrich property details for property_id: {property_id}")
            
            try:
                async with session.post(url, headers=self.headers, json={"property_id": property_id}) as r:
                    if r.status == 200:
                        data = await r.json()
                        # Check the structure based on the provided GPT-4o response hint
                        detail_data = data.get("data", {}).get("home", {})
                        
                        if detail_data:
                            # Update days_on_market - check tracking_fields first, then direct, then list_date
                            dom = detail_data.get("tracking_fields", {}).get("days_on_market")
                            if dom is None:
                                dom = detail_data.get("days_on_market") # Some markets expose it here

                            if dom is None and (list_date_str := detail_data.get("list_date")):
                                try:
                                    # compute it ourselves from list_date (ISO 8601 string)
                                    list_date_obj = datetime.fromisoformat(list_date_str[:10]).date()
                                    dom = (date.today() - list_date_obj).days
                                except Exception as e:
                                    logger.warning(f"Could not calculate days_on_market from list_date {list_date_str}: {e}")
                                    dom = 0 # Default to 0 if calculation fails

                            listing["days_on_market"] = dom if dom is not None else 0 # Ensure it's not None, default to 0

                            # Update square footage
                            sqft = detail_data.get("description", {}).get("sqft") or detail_data.get("building_size", {}).get("size")
                            if sqft is not None:
                                listing["area"] = sqft
                                
                            logger.info(f"Successfully enriched property_id {property_id}. Updated data: days_on_market={listing.get('days_on_market')}, area={listing.get('area')}")
                        else:
                             logger.warning(f"Detail data not found in response for property_id: {property_id}")

                    else:
                        logger.warning(f"Detail API request failed for property_id {property_id} with status {r.status}")

            except Exception as e:
                logger.error(f"Error fetching detail for property_id {property_id}: {e}")
            
        return listing

    def _format_property_details(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format detailed property information.
        """
        return {
            "property_id": property_data["property_id"],
            "address": property_data["location"]["address"]["line"],
            "city": property_data["location"]["address"]["city"],
            "state": property_data["location"]["address"]["state_code"],
            "zip_code": property_data["location"]["address"]["postal_code"],
            "price": property_data["list_price"],
            "beds": property_data["description"]["beds"],
            "baths": property_data["description"]["baths"],
            "square_feet": property_data["description"].get("sqft"),
            "property_type": property_data["property_type"],
            "listing_date": property_data["list_date"],
            "latitude": property_data["location"]["address"]["coordinate"]["lat"],
            "longitude": property_data["location"]["address"]["coordinate"]["lon"],
            "description": property_data["description"].get("text", ""),
            "features": property_data["description"].get("features", []),
            "photos": [photo["href"] for photo in property_data.get("photos", [])],
            "year_built": property_data["description"].get("year_built"),
            "lot_size": property_data["description"].get("lot_sqft")
        } 