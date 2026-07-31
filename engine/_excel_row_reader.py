# _excel_row_reader.py
# Standalone worker: reads the first sheet of an .xlsx file and prints it as
# JSON. Run as a subprocess (see import_account_engine._read_excel_rows)
# because parsing .xlsx with pandas inside a process that already has
# torch/OpenCV loaded (as app.py does at startup) segfaults on this
# environment — an isolated subprocess sidesteps the native library conflict.

import sys
import json
import pandas as pd


def main():
    file_path = sys.argv[1]
    xl = pd.ExcelFile(file_path)
    df = xl.parse(xl.sheet_names[0], dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    rows = df.where(pd.notna(df), None).to_dict(orient="records")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
