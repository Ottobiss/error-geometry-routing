"""Frame features for the router, one file per feature (or per family of features); see base.Feature.
To add a feature: a file with its class and an entry in FEATURES."""
from features.entropy import Entropy
from features.knn import KNN

FEATURES = {"entropy": Entropy, "knn": KNN}  # config name -> class
