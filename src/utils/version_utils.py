import json
import os

def get_model_version(default_version="2.4.0"):
    """
    Retrieves the clinical model version from the shared metadata artifact.
    """
    # Resolve project root (handles case where script is run from api/ or app/ subdirs)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metadata_path = os.path.join(base_dir, "models", "model_metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get("version", default_version)
        except Exception:
            return default_version
    return default_version
