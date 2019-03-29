import os
from pathlib import Path


AI_RESOURCE_PATH = Path(os.environ.get('AI_RESOURCE_PATH',
                                       './.pentago-ai-resouces'))
