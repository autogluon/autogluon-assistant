# src/autogluon/assistant/webui/backend/config.py

import os

class Config:
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = True # change it to false when depolying
