import csv

def test_input_files(path_to_mechanism:str, path_to_input:str, flame_type:str):
    with open(path_to_input, 'r') as file:
        csv_reader = csv.reader(file)

        if flame_type == 'stagnation':
            expected_headers = ['phi', 'phi Er', 'fuel', 'blend', 'T_in', 'T', 'U','P']
        elif flame_type == 'freely_prop_flame':
            expected_headers = ['phi', 'phi Er', 'fuel', 'blend', 'T_in', 'P', 'LBV', 'LBV Er']
        else:
            raise Exception('flame type not recognised')
        actual_headers = next(csv_reader)

        try:
            rows = list(csv_reader)  # Convert CSV reader to a list of rows
            cleaned_rows = []

            for row in rows:
                if all(cell.strip() == '' for cell in row):
                    continue  # Skip row if all cells are empty
                elif any(cell.strip() == '' for cell in row):
                    raise ValueError("Error: Found an empty cell in a row")  # Raise error if any single cell is empty
                else:
                    cleaned_rows.append(row)  # Append row if it has no empty cells

            # Write cleaned rows back to a new CSV file or update the existing one
            with open(path_to_input, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(cleaned_rows)

        except Exception as e:
            print("CSV format test failed:", str(e))
            return False

        for i, expected_row in enumerate(expected_headers):
            if i >= len(actual_headers):
                print("# CSV file doesn't have enough rows")
                return False
            if rows[i] != expected_row:
                print("Rows don't match the expected values")
                return False
            return False

    if not path_to_mechanism.endswith('.cti'):
        print("mech format file failed:")
        return False

    return True

