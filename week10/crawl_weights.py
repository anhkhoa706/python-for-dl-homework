import requests
from bs4 import BeautifulSoup
import pandas as pd

def crawl_table_to_excel(url: str, output_file: str) -> None:
    """
    Fetches the HTML at `url`, parses out the classification weights table,
    and writes it to `output_file` in Excel (.xlsx) format.

    Args:
        url: URL of the PyTorch Vision models page (must include the #table anchor).
        output_file: Path to the output .xlsx file.
    """
    # 1. Download page
    resp = requests.get(url)
    resp.raise_for_status()  # abort if we got an error

    # 2. Parse HTML
    soup = BeautifulSoup(resp.text, 'html.parser')

    # 3. Locate the table by anchor id
    header = soup.find(id='table-of-all-available-classification-weights')
    if header is None:
        raise RuntimeError("Could not find the table header with id 'table-of-all-available-classification-weights'")
    table = header.find_next('table')
    if table is None:
        raise RuntimeError("Could not find the table following the header")

    # 4. Use pandas to read the HTML table into a DataFrame
    df = pd.read_html(str(table), header=0)[0]

    # 5. Write DataFrame to Excel
    df.to_excel(output_file, index=False)
    print(f"✅ Successfully wrote table ({len(df)} rows, {len(df.columns)} columns) to '{output_file}'")

if __name__ == '__main__':
    URL = 'https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights'
    OUTPUT_FILE = 'classification_weights.xlsx'
    crawl_table_to_excel(URL, OUTPUT_FILE)
