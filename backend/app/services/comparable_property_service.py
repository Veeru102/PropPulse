import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from app.core.config import settings
from app.core.logging import loggers
from app.core.cache import comparable_cache, property_cache, market_cache
from app.services.data_collector import DataCollector

try:
    import faiss
except ImportError as _e:
    logger.error("FAISS library is not installed or failed to load: %s. Please ensure faiss-cpu is correctly installed (e.g., via conda install -c conda-forge faiss-cpu).", _e)
    faiss = None  # Set faiss to None if import fails

logger = loggers["ml"]


class ComparablePropertyService:
    """Service that provides comparable property search using latent embeddings
    produced by an AutoEncoder model and a FAISS similarity index.  If the queried
    property already exists in the index we directly use its stored embedding;
    otherwise we fall back to encoding a freshly-computed feature vector using the
    loaded AutoEncoder and feature scaler.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else self._find_latest_embedding_dir()
        logger.info(f"[ComparablePropertyService] Loading models from: {self.model_dir}")

        # DataCollector is needed for retrieving full property details after we
        # obtain similar IDs.  Import through ServiceManager to ensure singleton
        # behaviour and cached responses. Initialize this ALWAYS.
        from app.services.service_manager import ServiceManager
        self.data_collector: DataCollector = ServiceManager.get_data_collector()
        logger.info("[ComparablePropertyService] DataCollector initialized successfully")

        self.is_faiss_available = False  # Flag to track FAISS availability
        if faiss is None:
            logger.error("[ComparablePropertyService] FAISS module not loaded. ComparablePropertyService will operate in fallback-only mode.")
            logger.error("[ComparablePropertyService] To enable FAISS: conda install -c conda-forge faiss-cpu==1.7.4")
            return  # Skip loading artifacts if FAISS isn't available

        try:
            self._load_artifacts()
            self.is_faiss_available = True
            logger.info("[ComparablePropertyService] FAISS initialization SUCCESS - model, scaler, and index loaded")
            logger.info(f"[ComparablePropertyService] FAISS index contains {self.index.ntotal} property embeddings")
        except Exception as e:
            logger.error(f"[ComparablePropertyService] Failed to load FAISS artifacts: {e}. ComparablePropertyService will operate in fallback-only mode.")
            # is_faiss_available remains False

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def get_comparable_properties(self, property_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return *top_k* comparable properties for *property_id*.

        The return payload is a list of property dictionaries enriched with a
        ``similarity_score`` key (cosine similarity in latent space).
        """
        if not property_id:
            logger.error("get_comparable_properties called with empty property_id")
            return []

        # Check cache first
        cache_key = f"comp:{property_id}"
        if cached_results := comparable_cache.get(cache_key):
            logger.info(f"Cache HIT for comparable properties of {property_id}")
            return cached_results

        logger.info(f"Cache MISS - Finding comparable properties for property_id: {property_id}")

        # Only attempt FAISS search if it's available
        if not self.is_faiss_available:
            logger.warning(f"FAISS is not available. Skipping FAISS search for {property_id} and falling back to rules-based.")
            return [] # Indicate no FAISS results, triggering fallback in caller

        try:
            # First attempt: if the property is part of the index already, use the
            # stored vector (fast-path and avoids feature mismatches).
            if property_id in self._id_to_idx:
                logger.info(f"[Target Property] Property {property_id} found in FAISS index at index {self._id_to_idx[property_id]}")
                query_vector = self._reconstruct_vector(self._id_to_idx[property_id])
                
                # Validate the reconstructed vector
                if query_vector is None or not isinstance(query_vector, np.ndarray) or query_vector.size == 0:
                    logger.error(f"Failed to reconstruct valid vector for property {property_id} from FAISS index")
                    return []
                    
                logger.info(f"[Target Property] Successfully reconstructed vector with shape {query_vector.shape} for property {property_id}")
                logger.debug(f"[Target Property] Vector values (first 5): {query_vector.flatten()[:5]}")
            else:
                logger.info(f"Property {property_id} not in FAISS index, fetching details for encoding")
                # The property isn't in the index; fetch details and encode on the fly.
                property_data = await self.data_collector.get_property_by_id(property_id)
                if not property_data:
                    logger.error(f"Property {property_id} not found in data collector and not in FAISS index - cannot compute comparable properties.")
                    return []  # Return empty list instead of raising to allow graceful fallback
                    
                # Check if property data contains error status
                if property_data.get("status") == "unavailable" or property_data.get("error"):
                    logger.error(f"Property {property_id} is unavailable: {property_data.get('error', 'Unknown error')} and not in FAISS index - cannot compute comparable properties.")
                    return []

                logger.info(f"Successfully retrieved property data for {property_id}, encoding to vector")
                try:
                    query_vector = self._encode_property(property_data)
                    # Validate query_vector is a numpy array and not empty
                    if not isinstance(query_vector, np.ndarray) or query_vector.size == 0:
                        raise ValueError("Encoded query vector is invalid or empty")
                    logger.info(f"Successfully encoded property {property_id} to vector with shape {query_vector.shape}")
                except Exception as e:
                    logger.error(f"Failed to encode property {property_id} to vector: {str(e)}. This will trigger fallback.")
                    return []  # Return empty list to allow graceful fallback

            # Search FAISS index for most similar vectors (cosine similarity).
            logger.info(f"[FAISS Search] Searching FAISS index for {top_k} similar properties to {property_id}")
            logger.debug(f"[FAISS Search] Query vector shape: {query_vector.shape}, dtype: {query_vector.dtype}")
            logger.debug(f"[FAISS Search] FAISS index total vectors: {self.index.ntotal}, index dimension: {self.index.d}")
            
            try:
                scores, idxs = self.index.search(query_vector.astype("float32"), top_k + 1)  # +1 so we can drop self-match
                
                # Validate FAISS search results
                if scores is None or idxs is None:
                    logger.error(f"[FAISS Search] FAILED - search returned None results for property {property_id}")
                    return []
                    
                if scores.size == 0 or idxs.size == 0:
                    logger.warning(f"[FAISS Search] No similar properties found in FAISS index for property {property_id}")
                    return []
                
                scores = scores.flatten().tolist()
                idxs = idxs.flatten().tolist()
                
                # Validate that we have matching scores and indices
                if len(scores) != len(idxs):
                    logger.error(f"[FAISS Search] Mismatch between scores ({len(scores)}) and indices ({len(idxs)}) for property {property_id}")
                    return []
                    
                logger.info(f"[FAISS Search] SUCCESS - found {len(idxs)} candidates with similarity scores")
                logger.info(f"[FAISS Search] Top similarity scores: {[f'{s:.4f}' for s in scores[:3]]}")
                
                # Log the property IDs that FAISS found as similar
                similar_property_ids = []
                for idx in idxs:
                    if idx >= 0 and idx < self.index.ntotal:
                        comp_id = self._idx_to_id.get(idx)
                        if comp_id and property_id != comp_id:  # Exclude self-match
                            similar_property_ids.append(comp_id)
                
                logger.info(f"[FAISS Search] Similar property IDs found: {similar_property_ids[:10]}{'...' if len(similar_property_ids) > 10 else ''}")
                
            except Exception as e:
                logger.error(f"[FAISS Search] FAILED for property {property_id}: {str(e)}. This will trigger fallback.")
                return []

            results: List[Dict[str, Any]] = []
            failed_fetches = 0
            successful_fetches = 0
            
            logger.info(f"[Property Retrieval] Starting retrieval of {len(idxs)} candidate properties")

            for i, (score, idx) in enumerate(zip(scores, idxs)):
                logger.debug(f"[Property Retrieval] Processing candidate {i+1}/{len(idxs)}: FAISS_idx={idx}, score={score:.4f}")
                
                # Validate FAISS index
                if idx < 0 or idx >= self.index.ntotal:
                    logger.warning(f"[Property Retrieval] Invalid FAISS index {idx} (out of range 0-{self.index.ntotal-1}). Skipping.")
                    continue
                
                # Skip the query property itself if it's in the results
                if property_id in self._id_to_idx and idx == self._id_to_idx[property_id]:
                    logger.debug(f"[Property Retrieval] Skipping self-match for property {property_id} at FAISS index {idx}")
                    continue

                comp_id = self._idx_to_id.get(idx)
                if comp_id is None:
                    logger.warning(f"[Property Retrieval] Could not find property_id for FAISS index {idx} - index mapping may be corrupted")
                    continue  # Should not happen
                    
                # Validate comp_id is not empty
                if not comp_id or not isinstance(comp_id, str):
                    logger.warning(f"[Property Retrieval] Invalid property_id '{comp_id}' for FAISS index {idx}. Skipping.")
                    continue

                logger.debug(f"[Property Retrieval] Fetching comparable property {comp_id} (FAISS_idx={idx}, similarity={score:.4f})")
                try:
                    comp_data = await self.data_collector.get_property_by_id(comp_id)
                    if not comp_data:
                        logger.warning(f"[Property Retrieval] Property {comp_id} could not be fetched from data collector. Skipping.")
                        failed_fetches += 1
                        continue

                    # Validate the property data has required fields
                    if not isinstance(comp_data, dict) or not comp_data.get("property_id"):
                        logger.warning(f"[Property Retrieval] Property {comp_id} returned invalid data: {type(comp_data)}. Skipping.")
                        failed_fetches += 1
                        continue
                        
                    # Check if property data contains error status
                    if comp_data.get("status") == "unavailable" or comp_data.get("error"):
                        logger.warning(f"[Property Retrieval] Property {comp_id} is unavailable: {comp_data.get('error', 'Unknown error')}. Skipping.")
                        failed_fetches += 1
                        continue

                    comp_data = dict(comp_data)  # Shallow copy so we don't mutate cache
                    comp_data["similarity_score"] = float(score)
                    results.append(comp_data)
                    successful_fetches += 1
                    logger.debug(f"[Property Retrieval] SUCCESS - added comparable property {comp_id} (similarity={score:.4f})")

                    if len(results) >= top_k:
                        logger.debug(f"[Property Retrieval] Reached target count of {top_k} properties, stopping search")
                        break

                except Exception as e:
                    logger.error(f"[Property Retrieval] Error fetching comparable property {comp_id}: {str(e)}. Skipping.")
                    failed_fetches += 1
                    continue

            logger.info(f"[FAISS Search Complete] Property {property_id}: "
                       f"{successful_fetches} successful, {failed_fetches} failed fetches, "
                       f"{len(results)} total results returned from FAISS.")
            
            # Log details of final results for debugging
            for i, result in enumerate(results):
                logger.debug(f"[Final Results] {i+1}. property_id={result.get('property_id')}, "
                           f"similarity={result.get('similarity_score', 0):.4f}, "
                           f"address={result.get('address', 'N/A')}")

            if len(results) == 0:
                logger.warning(f"[FAISS Search Complete] No comparable properties could be fetched for {property_id}. "
                             f"This will trigger fallback to rules-based approach.")
                
                # Provide detailed explanation of what happened
                if failed_fetches > 0:
                    logger.info(f"[FAISS Analysis] FAISS found {failed_fetches} similar properties, but all failed to fetch current data.")
                    logger.info(f"[FAISS Analysis] This suggests the similar properties are historical listings no longer available.")
                    logger.info(f"[FAISS Analysis] This is expected behavior when FAISS index contains older property data.")
                else:
                    logger.info(f"[FAISS Analysis] No similar properties were found in the FAISS index for property {property_id}.")
            else:
                # Cache successful results
                cache_key = f"comp:{property_id}"
                comparable_cache.set(cache_key, results)  # 24h TTL by default
                logger.info(f"Cached comparable properties for {property_id}")

            return results

        except Exception as e:
            logger.error(f"Unexpected error during FAISS comparable property search for {property_id}: {str(e)}. "
                         f"This will trigger fallback to rules-based approach.")
            return []  # Return empty list to allow graceful fallback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_latest_embedding_dir(self) -> Path:
        """Return the most recently-modified subdirectory in *settings.MODEL_DIR*
        that contains the three required artifacts: autoencoder.pt,
        feature_scaler.pkl, and property_embeddings.faiss.
        """
        candidate_dirs = [p for p in Path(settings.MODEL_DIR).iterdir() if p.is_dir()]
        candidate_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for d in candidate_dirs:
            if (
                (d / "autoencoder.pt").exists()
                and (d / "feature_scaler.pkl").exists()
                and (d / "property_embeddings.faiss").exists()
            ):
                return d
        raise FileNotFoundError("Unable to locate a model directory containing FAISS index artefacts.")

    def _load_artifacts(self):
        # --------------------------
        # 1. Load Feature Scaler
        # --------------------------
        scaler_path = self.model_dir / "feature_scaler.pkl"
        try:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        except Exception as pickle_err:
            # Fallback for joblib-compressed files (common when using scikit-learn dump)
            try:
                import joblib  # import lazily to avoid hard dependency if unnecessary
                self.scaler = joblib.load(scaler_path)
                logger.warning(
                    "StandardScaler loaded via joblib due to pickle load failure (%s). Consider saving plain pickles for smaller footprints.",
                    pickle_err,
                )
            except Exception as joblib_err:
                raise RuntimeError(
                    f"Failed to load feature scaler from {scaler_path}: pickle_error={pickle_err} joblib_error={joblib_err}"
                ) from joblib_err

        # --------------------------
        # 2. Load / Rebuild AutoEncoder
        # --------------------------
        autoencoder_path = self.model_dir / "autoencoder.pt"
        from app.models.autoencoder import AutoEncoder  # local import to avoid cycles

        def _try_load(weights_only: bool):
            import inspect
            kwargs = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                kwargs["weights_only"] = weights_only
            return torch.load(str(autoencoder_path), **kwargs)

        try:
            obj = _try_load(True)
        except Exception as first_err:
            # The checkpoint was saved with the class reference as __main__.AutoEncoder
            # but when loading under uvicorn, __main__ is uvicorn's module.
            # We need to inject our AutoEncoder class into the __main__ namespace
            # so torch.load can find it.
            import sys
            import types
            
            # Create a fake __main__ module if needed or use existing
            if '__main__' not in sys.modules:
                main_module = types.ModuleType('__main__')
                sys.modules['__main__'] = main_module
            else:
                main_module = sys.modules['__main__']
            
            # Inject AutoEncoder into __main__
            setattr(main_module, 'AutoEncoder', AutoEncoder)
            
            # Also register as safe global for PyTorch 2.6+
            try:
                torch.serialization.add_safe_globals([AutoEncoder])
            except AttributeError:
                pass  # Older PyTorch versions don't have this
            
            try:
                obj = _try_load(False)
            except Exception as second_err:
                raise RuntimeError(
                    f"Failed to load autoencoder checkpoint {autoencoder_path}: {first_err} // {second_err}"
                ) from second_err

        if isinstance(obj, dict):
            # Got a state-dict → reconstruct model using scaler mean length if available
            inferred_dim = len(self.scaler.mean_) if hasattr(self.scaler, "mean_") else None
            if inferred_dim is None:
                # fallback to weight inspection
                first_key = next(k for k in obj.keys() if k.endswith("encoder.0.weight"))
                inferred_dim = obj[first_key].shape[1]
            self.model = AutoEncoder(input_dim=inferred_dim)
            self.model.load_state_dict(obj, strict=False)
        else:
            self.model = obj  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # 3. Model ready – derive input & latent dimensionalities *before* we
        #    reference them in any log statements.
        # ------------------------------------------------------------------
        self.model.eval()

        self._input_dim = (
            len(self.scaler.mean_)
            if hasattr(self.scaler, "mean_")
            else self.model.encoder[0].in_features  # type: ignore[attr-defined]
        )
        self._latent_dim = self.model.encoder[-1].out_features  # type: ignore[arg-type]

        logger.info(
            "AutoEncoder model loaded and ready for inference (latent_dim=%s, input_dim=%s)",
            self._latent_dim,
            self._input_dim,
        )

        # FAISS index
        index_path = self.model_dir / "property_embeddings.faiss"
        try:
            self.index = faiss.read_index(str(index_path))
            logger.info("FAISS index loaded with %s vectors and dimension %s", self.index.ntotal, self.index.d)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from {index_path}: {e}") from e

        # Property ID map (property_id -> index)
        id_map_path = self.model_dir / "property_id_map.json"
        with open(id_map_path, "r") as f:
            pid_to_idx = json.load(f)
        # Keys in JSON may not be strings – normalise to str for consistency
        self._id_to_idx: Dict[str, int] = {str(pid): int(idx) for pid, idx in pid_to_idx.items()}
        self._idx_to_id: Dict[int, str] = {idx: pid for pid, idx in self._id_to_idx.items()}

    def _reconstruct_vector(self, idx: int) -> np.ndarray:
        """Return the embedding vector stored in the FAISS index at *idx* as a
        `(1, latent_dim)` numpy array, already ℓ2-normalised.
        """
        vec = self.index.reconstruct(idx)
        return np.asarray(vec, dtype="float32").reshape(1, -1)

    # ----------------------- feature engineering -----------------------
    def _encode_property(self, prop: Dict[str, Any]) -> np.ndarray:
        """Compute a latent embedding for *prop* (formatted dictionary from
        DataCollector) compatible with the FAISS index.
        """
        logger.debug(f"[_encode_property] Starting encoding for property with keys: {list(prop.keys())}")
        
        # Validate input property data
        if not prop or not isinstance(prop, dict):
            raise ValueError(f"Invalid property data: expected dict, got {type(prop)}")
        
        # Log key property fields for debugging
        logger.debug(f"[_encode_property] Property details - price: {prop.get('price')}, sqft: {prop.get('square_feet')}, beds: {prop.get('beds')}, baths: {prop.get('baths')}")
        
        # Build a feature vector.  We keep the ordering stable to match the
        # scaler that was used during training.  If the scaler expects a larger
        # dimensionality we pad with zeros.  The chosen core features cover the
        # most universally-available numeric attributes, reducing the likelihood
        # of NaNs.
        feature_vals: List[float] = [
            float(prop.get("price", 0)),
            float(prop.get("square_feet", 0)),
            float(prop.get("beds", 0)),
            float(prop.get("baths", 0)),
            float(prop.get("year_built", 0)),
            float(prop.get("lot_size", 0)),
            float(prop.get("latitude", 0)),
            float(prop.get("longitude", 0)),
        ]
        
        logger.debug(f"[_encode_property] Raw feature vector (8 values): {feature_vals}")

        # Pad or truncate to *self._input_dim*
        original_len = len(feature_vals)
        if len(feature_vals) < self._input_dim:
            feature_vals.extend([0.0] * (self._input_dim - len(feature_vals)))
            logger.debug(f"[_encode_property] Padded feature vector from {original_len} to {len(feature_vals)} dimensions")
        elif len(feature_vals) > self._input_dim:
            feature_vals = feature_vals[: self._input_dim]
            logger.debug(f"[_encode_property] Truncated feature vector from {original_len} to {len(feature_vals)} dimensions")

        features_np = np.asarray(feature_vals, dtype=np.float32).reshape(1, -1)
        logger.debug(f"[_encode_property] Feature array shape before scaling: {features_np.shape}")
        
        # Standardise
        try:
            features_np = self.scaler.transform(features_np)
            logger.debug(f"[_encode_property] Feature scaling SUCCESS - shape: {features_np.shape}")
        except Exception as e:
            logger.warning(f"[_encode_property] Scaler.transform failed (shape mismatch {features_np.shape}): {e}. Attempting to auto-trim/pad.")
            expected = getattr(self.scaler, "mean_", np.zeros(features_np.shape[1])).shape[0]
            if features_np.shape[1] > expected:
                features_np = features_np[:, :expected]
                logger.debug(f"[_encode_property] Trimmed features to {expected} dimensions")
            else:
                features_np = np.pad(features_np, ((0, 0), (0, expected - features_np.shape[1])), constant_values=0)
                logger.debug(f"[_encode_property] Padded features to {expected} dimensions")
            features_np = self.scaler.transform(features_np)
            logger.debug(f"[_encode_property] Feature scaling SUCCESS after adjustment - shape: {features_np.shape}")

        # Encode into latent space
        logger.debug(f"[_encode_property] Encoding to latent space using autoencoder...")
        try:
            with torch.no_grad():
                tensor_in = torch.tensor(features_np, dtype=torch.float32)
                logger.debug(f"[_encode_property] Input tensor shape: {tensor_in.shape}")
                _, z = self.model(tensor_in)
                logger.debug(f"[_encode_property] Autoencoder output shape: {z.shape}")
            z_np = z.numpy()
            logger.debug(f"[_encode_property] Converted to numpy array shape: {z_np.shape}")
            
            # ℓ2 normalise for cosine similarity
            norms = np.linalg.norm(z_np, axis=1, keepdims=True)
            z_np = z_np / np.clip(norms, a_min=1e-8, a_max=None)
            logger.debug(f"[_encode_property] L2 normalization complete - final embedding shape: {z_np.shape}")
            logger.info(f"[_encode_property] SUCCESS - property encoded to {z_np.shape} latent vector")
            
            return z_np
        except Exception as e:
            logger.error(f"[_encode_property] Autoencoder encoding FAILED: {str(e)}")
            raise 