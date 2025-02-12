"""
Script to update imports after project restructuring.
"""

import os
import re
from pathlib import Path

# Define import mappings
IMPORT_MAPPINGS = {
    "from src.data_loader": "from src.data.data_loader",
    "from src.forex_data_loader": "from src.data.forex_data_loader",
    "from src.fetch_intraday_data": "from src.data.intraday_data_loader",
    "from src.eigenportfolio_analyzer": "from src.analysis.eigenportfolio",
    "from src.eigenportfolio_builder": "from src.analysis.eigenportfolio",
    "from src.arbitrage_signal_detector": "from src.analysis.arbitrage_detector",
    "from src.transfer_entropy": "from src.utils.transfer_entropy",
    "from src.execution_engine": "from src.trading.execution_engine",
    "from src.paper_trading": "from src.trading.paper_trading",
    "from src.realtime_strategy": "from src.trading.realtime_strategy",
    "from src.backtester": "from src.backtesting.backtester",
    "from src.paper_trading_simulation": "from src.simulation.paper_trading_simulation",
    "from src.config": "from src.config.settings",
    "from src.ml_model_trainer": "from src.models.ml_model",
    "from src.train_rl_strategy": "from src.models.rl_model",
    "from src.websocket_client": "from src.market_data.websocket_client",
    "from src.tiingo_forex_client": "from src.market_data.tiingo_client",
}


def update_imports(file_path: str) -> None:
    """Update imports in a Python file."""
    try:
        # Try UTF-8 first
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Try Latin-1 if UTF-8 fails
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return

    # Only process our project files
    if not any(old_import in content for old_import in IMPORT_MAPPINGS.keys()):
        return

    # Update imports
    for old_import, new_import in IMPORT_MAPPINGS.items():
        content = content.replace(old_import, new_import)

    try:
        # Write updated content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated imports in {file_path}")
    except Exception as e:
        print(f"Error writing {file_path}: {str(e)}")


def main():
    # Get all Python files in the project
    project_root = Path(__file__).parent.parent
    python_files = []
    for root, _, files in os.walk(project_root):
        if "venv" in root or ".git" in root:  # Skip venv and git directories
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    # Update imports in each file
    for file_path in python_files:
        update_imports(file_path)


if __name__ == "__main__":
    main()
