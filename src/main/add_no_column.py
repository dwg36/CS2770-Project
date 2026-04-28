import csv
import sys
import os

def add_no_column(input_csv_path, output_csv_path=None):
    # 출력 파일 경로가 없으면 자동 생성
    if output_csv_path is None:
        base, ext = os.path.splitext(input_csv_path)
        output_csv_path = f"{base}_with_no{ext}"

    with open(input_csv_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 첫 줄 (header)
        header = next(reader)
        new_header = ["no"] + header
        writer.writerow(new_header)

        # 데이터 rows
        for idx, row in enumerate(reader):
            new_row = [idx] + row
            writer.writerow(new_row)

    print(f"Done! Output saved to: {output_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_no_column.py input.csv [output.csv]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    add_no_column(input_csv, output_csv)
