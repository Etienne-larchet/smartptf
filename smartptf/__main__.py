import argparse

from dotenv import load_dotenv

from .config import DATA_DIR
from .logging_config import configure_logging

load_dotenv()
configure_logging()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default=f"{DATA_DIR}/sp500_compo_until_2025-03-10.csv",
        help="Path to the CSV file containing S&P 500 composition",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Reference date for Index composition (YYYY-MM-DD)",
    )
     # See exemples of use in test folder

if __name__ == "__main__":
    main()
