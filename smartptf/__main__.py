import argparse

from .config import DATA_DIR
from .logging_config import configure_logging
from .utils.Load import Indice

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
    args = parser.parse_args()

    sp500 = Indice(name='SP500', csv_compo_path=args.csv_path, date_end=args.date, period="4y")
    sp500.load_from_csv()
    print(sp500.open)


if __name__ == "__main__":
    main()
