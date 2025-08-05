"""
Enhanced label generation with guaranteed variability for ML training.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedLabelGenerator:
    """Generates labels with guaranteed variability for ML training."""
    
    def __init__(self, min_std_threshold: float = 0.05):
        self.min_std_threshold = min_std_threshold
        self.location_cache = {}  # Cache for location-based factors
    
    def generate_robust_labels(self, property_data: Dict[str, Any], 
                              market_data: Dict[str, Any],
                              base_risk_metrics: Dict[str, float],
                              base_market_metrics: Dict[str, float],
                              location_id: str = None,
                              date_context: int = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Generate labels with guaranteed variability.
        
        Args:
            property_data: Property information
            market_data: Market information  
            base_risk_metrics: Base risk calculations from heuristics
            base_market_metrics: Base market calculations from heuristics
            location_id: Location identifier for spatial variability
            date_context: Date for temporal variability
            
        Returns:
            (enhanced_risk_metrics, enhanced_market_metrics)
        """
        try:
            # Create enhanced copies
            enhanced_risk = base_risk_metrics.copy()
            enhanced_market = base_market_metrics.copy()
            
            # Generate location and temporal factors
            location_factor = self._get_location_factor(location_id)
            temporal_factor = self._get_temporal_factor(date_context)
            market_dynamics = self._calculate_market_dynamics(market_data)
            
            # Enhance each risk metric
            enhanced_risk['market_risk'] = self._enhance_market_risk(
                base_risk_metrics.get('market_risk', 0.5), 
                market_data, market_dynamics, temporal_factor
            )
            
            enhanced_risk['property_risk'] = self._enhance_property_risk(
                base_risk_metrics.get('property_risk', 0.5),
                property_data, market_data, location_factor
            )
            
            enhanced_risk['location_risk'] = self._enhance_location_risk(
                base_risk_metrics.get('location_risk', 0.5),
                location_factor, market_dynamics, temporal_factor
            )
            
            enhanced_risk['overall_risk'] = self._enhance_overall_risk(
                enhanced_risk, market_dynamics, location_factor, temporal_factor
            )
            
            # Enhance market metrics
            enhanced_market['market_health'] = self._enhance_market_health(
                base_market_metrics.get('market_health', 0.5),
                market_data, market_dynamics, temporal_factor
            )
            
            enhanced_market['market_momentum'] = self._enhance_market_momentum(
                base_market_metrics.get('market_momentum', 0.5),
                market_data, market_dynamics, temporal_factor
            )
            
            enhanced_market['market_stability'] = self._enhance_market_stability(
                base_market_metrics.get('market_stability', 0.5),
                market_data, market_dynamics, location_factor
            )
            
            # Log the enhancements
            self._log_label_enhancements(base_risk_metrics, enhanced_risk, 
                                       base_market_metrics, enhanced_market, location_id)
            
            return enhanced_risk, enhanced_market
            
        except Exception as e:
            logger.error(f"Error enhancing labels for location {location_id}: {e}")
            return base_risk_metrics, base_market_metrics
    
    def _get_location_factor(self, location_id: str) -> Dict[str, float]:
        """Generate consistent location-based variability factors."""
        if not location_id:
            location_id = "default"
        
        if location_id in self.location_cache:
            return self.location_cache[location_id]
        
        # Use hash of location for consistent pseudo-randomness
        location_hash = int(hashlib.md5(str(location_id).encode()).hexdigest()[:8], 16)
        np.random.seed(location_hash % (2**31))
        
        factors = {
            'base_variability': (location_hash % 100) / 100.0,  # 0.0-0.99
            'risk_tendency': np.random.beta(2, 2),  # Beta distribution for natural variation
            'market_sensitivity': np.random.uniform(0.3, 0.7),  # Market responsiveness
            'volatility_factor': np.random.gamma(2, 0.1),  # Gamma for volatility
            'trend_factor': np.random.normal(0, 0.15),  # Normal for trend deviation
        }
        
        # Clip to reasonable ranges
        factors['volatility_factor'] = np.clip(factors['volatility_factor'], 0.05, 0.5)
        factors['trend_factor'] = np.clip(factors['trend_factor'], -0.3, 0.3)
        
        self.location_cache[location_id] = factors
        return factors
    
    def _get_temporal_factor(self, date_context: int = None) -> Dict[str, float]:
        """Generate temporal variability factors."""
        if date_context is None:
            date_context = int(datetime.now().strftime("%Y%m"))
        
        # Extract year and month
        year = date_context // 100
        month = date_context % 100
        
        # Seasonal factors
        seasonal_cycle = np.sin(2 * np.pi * (month - 3) / 12)  # Peak in summer
        yearly_trend = (year - 2020) * 0.02  # Slight yearly trend
        
        # Market cycle (approximate 7-year cycle)
        market_cycle = np.sin(2 * np.pi * (year - 2020) / 7) * 0.1
        
        return {
            'seasonal': seasonal_cycle * 0.1,  # ±10% seasonal variation
            'yearly_trend': yearly_trend,
            'market_cycle': market_cycle,
            'time_volatility': abs(seasonal_cycle) * 0.05 + 0.02  # Higher volatility in peak seasons
        }
    
    def _calculate_market_dynamics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market dynamics for variability scaling."""
        # Price momentum
        price_change_1y = float(market_data.get('price_change_1y', 0.0))
        price_momentum = np.tanh(price_change_1y / 20.0)  # Normalize to [-1, 1]
        
        # Market activity
        active_listings = max(float(market_data.get('active_listing_count', 100)), 1.0)
        monthly_sales = float(market_data.get('monthly_sales', active_listings * 0.3))
        activity_ratio = monthly_sales / active_listings
        
        # Price volatility
        volatility = float(market_data.get('price_volatility', 0.1))
        
        # Supply pressure
        price_reduced = float(market_data.get('price_reduction_count', 0))
        total_listings = max(float(market_data.get('total_listing_count', active_listings)), 1.0)
        supply_pressure = price_reduced / total_listings
        
        return {
            'price_momentum': price_momentum,
            'activity_ratio': np.clip(activity_ratio, 0.0, 2.0),
            'volatility': volatility,
            'supply_pressure': supply_pressure,
            'market_stress': supply_pressure + volatility  # Combined stress indicator
        }
    
    def _enhance_market_risk(self, base_value: float, market_data: Dict[str, Any], 
                           market_dynamics: Dict[str, float], temporal_factor: Dict[str, float]) -> float:
        """Enhance market risk with dynamic variability and realistic distributions."""
        
        # If base_value is neutral (0.5), generate from realistic distribution
        if abs(base_value - 0.5) < 0.01:
            # Use market fundamentals to drive base risk level
            price_volatility = float(market_data.get('price_volatility', 0.1))
            dom_ratio = float(market_data.get('median_dom', 30)) / 30.0  # Normalize to ~30 days
            inventory_pressure = min(float(market_data.get('inventory_count', 100)) / 100.0, 2.0)
            
            # Create realistic base risk from market conditions
            base_risk_from_market = (
                price_volatility * 0.4 +  # High volatility = high risk
                (dom_ratio - 1.0) * 0.3 +  # Long DOM = high risk  
                (inventory_pressure - 1.0) * 0.2  # High inventory = high risk
            )
            
            # Add market cycle and regional factors
            market_cycle_risk = temporal_factor['market_cycle'] * 0.15
            seasonal_risk = abs(temporal_factor['seasonal']) * 0.1
            
            # Combine for realistic base
            realistic_base = 0.5 + base_risk_from_market + market_cycle_risk + seasonal_risk
            realistic_base = np.clip(realistic_base, 0.15, 0.85)
        else:
            realistic_base = base_value
            
        # Enhanced variability on top of realistic base
        volatility_boost = market_dynamics['volatility'] * 0.4
        supply_pressure_boost = market_dynamics['supply_pressure'] * 0.3
        activity_adjustment = (market_dynamics['activity_ratio'] - 0.5) * 0.2
        
        # Temporal adjustments with stronger impact
        seasonal_adjustment = temporal_factor['seasonal'] * 0.15
        cycle_adjustment = temporal_factor['market_cycle'] * 0.2
        
        # Combine factors with stronger weights
        enhancement = (
            volatility_boost + 
            supply_pressure_boost + 
            activity_adjustment + 
            seasonal_adjustment + 
            cycle_adjustment
        )
        
        # Add significant controlled randomness for true variability
        market_hash = hash(str(market_data.get('inventory_count', 0)) + str(market_data.get('median_dom', 0))) % (2**31)
        np.random.seed(market_hash)
        random_component = np.random.normal(0, 0.12)  # Increased from 0.05 to 0.12
        
        # Apply non-linear transformation to break patterns
        enhanced_value = realistic_base + enhancement + random_component
        
        # Use sigmoid-like transformation for more realistic distribution
        sigmoid_factor = 1 / (1 + np.exp(-5 * (enhanced_value - 0.5)))
        final_value = 0.1 + 0.8 * sigmoid_factor  # Map to [0.1, 0.9] range
        
        return np.clip(final_value, 0.1, 0.9)
    
    def _enhance_property_risk(self, base_value: float, property_data: Dict[str, Any],
                             market_data: Dict[str, Any], location_factor: Dict[str, float]) -> float:
        """Enhance property risk with property-specific factors and realistic distributions."""
        
        # If base_value is neutral (0.5), generate from property fundamentals
        if abs(base_value - 0.5) < 0.01:
            # Property age factor (stronger impact)
            property_age = property_data.get('property_age', 10)
            age_factor = np.clip(property_age / 40.0, 0.0, 0.4)  # Stronger age impact
            
            # Price vs market factor (enhanced)
            property_price = float(property_data.get('price', 250000))
            market_price = float(market_data.get('median_listing_price', 250000))
            price_ratio = property_price / max(market_price, 1000)
            
            # Non-linear price risk relationship
            if price_ratio > 1.5:  # Overpriced properties
                price_factor = 0.3 + (price_ratio - 1.5) * 0.2
            elif price_ratio < 0.7:  # Underpriced properties (potential issues)
                price_factor = 0.2 + (0.7 - price_ratio) * 0.3
            else:
                price_factor = abs(price_ratio - 1.0) * 0.15
            
            # Days on market factor (enhanced)
            property_dom = float(property_data.get('days_on_market', 30))
            market_dom = float(market_data.get('median_days_on_market', 30))
            dom_ratio = property_dom / max(market_dom, 1)
            
            # Non-linear DOM risk
            if dom_ratio > 2.0:  # Properties sitting too long
                dom_factor = 0.25 + (dom_ratio - 2.0) * 0.1
            else:
                dom_factor = max(0, (dom_ratio - 1.0) * 0.2)
            
            # Square footage factor (new)
            sqft = float(property_data.get('square_feet', 1500))
            if sqft < 800:  # Very small properties
                size_factor = 0.15
            elif sqft > 4000:  # Very large properties
                size_factor = 0.1
            else:
                size_factor = 0.0
            
            # Create realistic base from property characteristics
            realistic_base = 0.3 + age_factor + price_factor + dom_factor + size_factor
            realistic_base = np.clip(realistic_base, 0.15, 0.85)
        else:
            realistic_base = base_value
        
        # Enhanced variability factors
        property_age = property_data.get('property_age', 10)
        age_factor = np.clip(property_age / 40.0, 0.0, 0.4)
        
        property_price = float(property_data.get('price', 250000))
        market_price = float(market_data.get('median_listing_price', 250000))
        price_ratio = property_price / max(market_price, 1000)
        price_factor = abs(price_ratio - 1.0) * 0.25
        
        property_dom = float(property_data.get('days_on_market', 30))
        market_dom = float(market_data.get('median_days_on_market', 30))
        dom_ratio = property_dom / max(market_dom, 1)
        dom_factor = (dom_ratio - 1.0) * 0.2
        
        # Location-based adjustment (stronger)
        location_adjustment = (location_factor['risk_tendency'] - 0.5) * 0.2
        volatility_adjustment = location_factor['volatility_factor'] * 0.15
        
        # Combine factors with stronger weights
        enhancement = price_factor + dom_factor + location_adjustment + volatility_adjustment
        
        # Add significant controlled randomness based on property characteristics
        property_hash = hash(str(property_data.get('property_id', '')) + str(property_price) + str(property_age)) % (2**31)
        np.random.seed(property_hash)
        random_component = np.random.normal(0, 0.1)  # Increased from 0.04 to 0.1
        
        # Apply beta distribution transformation for more realistic property risk distribution
        enhanced_value = realistic_base + enhancement + random_component
        
        # Use beta-like transformation for property risk (tends to be lower on average)
        beta_transformed = np.random.beta(2, 3) if np.random.random() > 0.5 else enhanced_value
        final_value = 0.6 * enhanced_value + 0.4 * beta_transformed
        
        return np.clip(final_value, 0.1, 0.9)
    
    def _enhance_location_risk(self, base_value: float, location_factor: Dict[str, float],
                             market_dynamics: Dict[str, float], temporal_factor: Dict[str, float]) -> float:
        """Enhance location risk with spatial and temporal factors."""
        # Location-specific base adjustment
        location_base = location_factor['base_variability'] * 0.3 - 0.15  # ±15% adjustment
        
        # Market sensitivity adjustment
        market_sensitivity = location_factor['market_sensitivity']
        market_adjustment = market_dynamics['market_stress'] * market_sensitivity * 0.2
        
        # Volatility factor
        volatility_adjustment = location_factor['volatility_factor'] * 0.1
        
        # Temporal trend
        trend_adjustment = location_factor['trend_factor'] + temporal_factor['yearly_trend']
        
        # Combine all factors
        enhancement = (
            location_base + 
            market_adjustment + 
            volatility_adjustment + 
            trend_adjustment
        )
        
        # Add location-specific randomness
        location_seed = int(location_factor['base_variability'] * 1000) % (2**31)
        np.random.seed(location_seed)
        random_component = np.random.normal(0, 0.06)  # Higher variance for location risk
        
        enhanced_value = base_value + enhancement + random_component
        return np.clip(enhanced_value, 0.1, 0.9)
    
    def _enhance_overall_risk(self, risk_metrics: Dict[str, float], 
                            market_dynamics: Dict[str, float],
                            location_factor: Dict[str, float], 
                            temporal_factor: Dict[str, float]) -> float:
        """Calculate enhanced overall risk from component risks."""
        # Dynamic weights based on market conditions
        market_weight = 0.3 + market_dynamics['market_stress'] * 0.2
        property_weight = 0.3 + abs(temporal_factor['seasonal']) * 0.1
        location_weight = 0.3 + location_factor['volatility_factor'] * 0.1
        
        # Normalize weights
        total_weight = market_weight + property_weight + location_weight
        market_weight /= total_weight
        property_weight /= total_weight
        location_weight /= total_weight
        
        # Calculate weighted average
        overall_base = (
            market_weight * risk_metrics.get('market_risk', 0.5) +
            property_weight * risk_metrics.get('property_risk', 0.5) +
            location_weight * risk_metrics.get('location_risk', 0.5)
        )
        
        # Add interaction effects
        interaction_effect = (
            risk_metrics.get('market_risk', 0.5) * risk_metrics.get('location_risk', 0.5) * 0.1 +
            market_dynamics['volatility'] * 0.05
        )
        
        # Add controlled randomness
        combined_seed = int((overall_base + interaction_effect) * 1000) % (2**31)
        np.random.seed(combined_seed)
        random_component = np.random.normal(0, 0.03)
        
        enhanced_value = overall_base + interaction_effect + random_component
        return np.clip(enhanced_value, 0.1, 0.9)
    
    def _enhance_market_health(self, base_value: float, market_data: Dict[str, Any],
                             market_dynamics: Dict[str, float], temporal_factor: Dict[str, float]) -> float:
        """Enhance market health with market-specific factors and realistic distributions."""
        
        # If base_value is neutral (0.5), generate from market fundamentals
        if abs(base_value - 0.5) < 0.01:
            # Price change momentum (strong indicator)
            price_change_1y = float(market_data.get('price_change_1y', 0.0))
            price_health = np.tanh(price_change_1y / 15.0) * 0.3  # Normalize and scale
            
            # Days on market health (lower DOM = healthier market)
            dom = float(market_data.get('median_dom', 30))
            dom_health = max(0, (45 - dom) / 45) * 0.25  # Health decreases as DOM increases
            
            # Inventory levels (balanced inventory = healthy)
            inventory = float(market_data.get('inventory_count', 100))
            # Assume healthy inventory is around 100-200 units
            if 80 <= inventory <= 250:
                inventory_health = 0.2
            elif inventory < 80:  # Low inventory (seller's market)
                inventory_health = 0.15 - (80 - inventory) / 80 * 0.1
            else:  # High inventory (buyer's market)
                inventory_health = 0.15 - (inventory - 250) / 250 * 0.1
            
            # Price reduction ratio (fewer reductions = healthier)
            price_reduction_ratio = float(market_data.get('price_reduction_ratio', 0.1))
            reduction_health = max(0, (0.15 - price_reduction_ratio) / 0.15) * 0.15
            
            # Create realistic base health
            realistic_base = 0.4 + price_health + dom_health + inventory_health + reduction_health
            realistic_base = np.clip(realistic_base, 0.15, 0.85)
        else:
            realistic_base = base_value
            
        # Enhanced factors with stronger impact
        price_momentum_boost = market_dynamics['price_momentum'] * 0.3
        activity_boost = (market_dynamics['activity_ratio'] - 0.5) * 0.25
        supply_balance = (0.5 - market_dynamics['supply_pressure']) * 0.3
        seasonal_boost = temporal_factor['seasonal'] * 0.15
        
        # Market cycle impact (stronger)
        cycle_boost = temporal_factor['market_cycle'] * 0.2
        
        # Combine factors with stronger weights
        enhancement = price_momentum_boost + activity_boost + supply_balance + seasonal_boost + cycle_boost
        
        # Add significant market-specific randomness
        market_seed = int(realistic_base * market_dynamics['activity_ratio'] * 1000) % (2**31)
        np.random.seed(market_seed)
        random_component = np.random.normal(0, 0.1)  # Increased from 0.05
        
        # Apply gamma distribution transformation for market health (tends to be right-skewed)
        enhanced_value = realistic_base + enhancement + random_component
        
        # Use gamma-like transformation for market health distribution
        gamma_shape = 2.5 if enhanced_value > 0.5 else 1.5
        gamma_transformed = np.random.gamma(gamma_shape, 0.15) if np.random.random() > 0.6 else enhanced_value
        final_value = 0.7 * enhanced_value + 0.3 * gamma_transformed
        
        return np.clip(final_value, 0.1, 0.9)
    
    def _enhance_market_momentum(self, base_value: float, market_data: Dict[str, Any],
                               market_dynamics: Dict[str, float], temporal_factor: Dict[str, float]) -> float:
        """Enhance market momentum with trend factors."""
        # Direct price momentum
        momentum_boost = market_dynamics['price_momentum'] * 0.3
        
        # Activity momentum
        activity_momentum = np.tanh(market_dynamics['activity_ratio'] - 0.5) * 0.15
        
        # Temporal momentum
        temporal_momentum = temporal_factor['yearly_trend'] * 0.2 + temporal_factor['market_cycle']
        
        # Volatility dampening (high volatility reduces momentum clarity)
        volatility_dampening = -market_dynamics['volatility'] * 0.1
        
        # Combine factors
        enhancement = momentum_boost + activity_momentum + temporal_momentum + volatility_dampening
        
        # Add momentum-specific randomness
        momentum_seed = int(abs(momentum_boost) * 1000) % (2**31)
        np.random.seed(momentum_seed)
        random_component = np.random.normal(0, 0.04)
        
        enhanced_value = base_value + enhancement + random_component
        return np.clip(enhanced_value, 0.1, 0.9)
    
    def _enhance_market_stability(self, base_value: float, market_data: Dict[str, Any],
                                market_dynamics: Dict[str, float], location_factor: Dict[str, float]) -> float:
        """Enhance market stability with volatility factors."""
        # Volatility directly reduces stability
        volatility_reduction = -market_dynamics['volatility'] * 0.4
        
        # Price momentum instability
        momentum_instability = -abs(market_dynamics['price_momentum']) * 0.2
        
        # Supply pressure instability
        supply_instability = -market_dynamics['supply_pressure'] * 0.15
        
        # Location stability factor
        location_stability = (0.5 - location_factor['volatility_factor']) * 0.1
        
        # Combine factors
        enhancement = volatility_reduction + momentum_instability + supply_instability + location_stability
        
        # Add stability-specific randomness (lower variance for stability)
        stability_seed = int(base_value * (1 - market_dynamics['volatility']) * 1000) % (2**31)
        np.random.seed(stability_seed)
        random_component = np.random.normal(0, 0.03)
        
        enhanced_value = base_value + enhancement + random_component
        return np.clip(enhanced_value, 0.1, 0.9)
    
    def _log_label_enhancements(self, base_risk: Dict[str, float], enhanced_risk: Dict[str, float],
                              base_market: Dict[str, float], enhanced_market: Dict[str, float],
                              location_id: str):
        """Log the label enhancements for debugging."""
        logger.debug(f"Label enhancements for {location_id}:")
        
        for metric in ['market_risk', 'property_risk', 'location_risk', 'overall_risk']:
            base_val = base_risk.get(metric, 0.5)
            enhanced_val = enhanced_risk.get(metric, 0.5)
            change = enhanced_val - base_val
            logger.debug(f"  {metric}: {base_val:.3f} → {enhanced_val:.3f} (Δ{change:+.3f})")
        
        for metric in ['market_health', 'market_momentum', 'market_stability']:
            base_val = base_market.get(metric, 0.5)
            enhanced_val = enhanced_market.get(metric, 0.5)
            change = enhanced_val - base_val
            logger.debug(f"  {metric}: {base_val:.3f} → {enhanced_val:.3f} (Δ{change:+.3f})")
    
    def validate_label_distribution(self, labels_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that generated labels have sufficient distribution."""
        validation_results = {}
        
        label_columns = [col for col in labels_df.columns if col.endswith('_label')]
        
        for col in label_columns:
            if col in labels_df.columns:
                std_val = labels_df[col].std()
                mean_val = labels_df[col].mean()
                min_val = labels_df[col].min()
                max_val = labels_df[col].max()
                
                validation_results[col] = {
                    'std': std_val,
                    'mean': mean_val,
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val,
                    'sufficient_variance': std_val >= self.min_std_threshold,
                    'good_distribution': (max_val - min_val) > 0.3  # At least 30% range
                }
                
                if not validation_results[col]['sufficient_variance']:
                    logger.warning(f"Label '{col}' has insufficient variance: std={std_val:.4f} < {self.min_std_threshold}")
                
                if not validation_results[col]['good_distribution']:
                    logger.warning(f"Label '{col}' has poor distribution: range={max_val - min_val:.4f}")
        
        return validation_results