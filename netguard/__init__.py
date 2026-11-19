"""netguard-ml: network anomaly detection with an Isolation Forest + Autoencoder ensemble."""

from netguard.ensemble import PredictionResult, RiskLevel
from netguard.predict import Predictor

__version__ = "0.1.0"

__all__ = ["PredictionResult", "RiskLevel", "Predictor", "__version__"]
