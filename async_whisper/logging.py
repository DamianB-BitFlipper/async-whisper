import logging

# Configure the logging module
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create a global logger
logger = logging.getLogger("async_whisper")
