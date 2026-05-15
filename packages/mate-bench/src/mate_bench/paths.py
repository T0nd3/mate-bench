from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir

CACHE_DIR = Path(user_cache_dir("mate-bench"))
DATA_DIR = Path(user_data_dir("mate-bench"))
CONFIG_DIR = Path(user_config_dir("mate-bench"))

MODELS_DIR = CACHE_DIR / "models"
TEST_SETS_DIR = CACHE_DIR / "test-sets"
RESULTS_DIR = DATA_DIR / "results"
