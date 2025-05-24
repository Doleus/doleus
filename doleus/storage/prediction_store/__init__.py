from doleus.storage.prediction_store.base import BasePredictionStore
from doleus.storage.prediction_store.classification import ClassificationPredictionStore
from doleus.storage.prediction_store.detection import DetectionPredictionStore

__all__ = [
    "BasePredictionStore",
    "ClassificationPredictionStore", 
    "DetectionPredictionStore",
] 