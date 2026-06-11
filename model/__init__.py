import os
from pathlib import Path

from dotenv import load_dotenv  # noqa: F401

import utils  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
CONFIG_PATH = os.environ.get("CONFIG_PATH")

config = utils.conv_yaml_dc(CONFIG_PATH)
