"""
Test API key loading.
"""

import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_key():
    """Test that the API key is loaded correctly."""
    load_dotenv()

    # Check if API key exists
    api_key = os.getenv("TIINGO_API_KEY")
    if api_key:
        # Only show first and last few characters for security
        masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        logger.info(f"API key found: {masked_key}")
        return True
    else:
        logger.error("TIINGO_API_KEY not found in environment variables")
        return False


if __name__ == "__main__":
    test_api_key()
