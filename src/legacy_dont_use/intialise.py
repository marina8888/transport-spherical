import csv

def check_input(mech:str, exp_results:str, flame_type:str):
    with open(exp_results, 'r') as file:
        csv_reader = csv.reader(file)
        actual_headers = next(csv_reader)
        expected_headers = []

        if flame_type == 'stagnation':
            expected_headers = ['phi', 'phi Er', 'fuel', 'blend', 'T_in', 'T', 'U','P']
        elif flame_type == 'freely_prop_0.5H2_0.5NH3':
            expected_headers = ['phi', 'phi Er', 'fuel', 'blend', 'T_in', 'P', 'LBV', 'LBV Er']
        else:
            raise Exception('flame type not recognised')

        # Check cells:
        try:
            rows = list(csv_reader)
            cleaned_rows = []

            for row in rows:
                if all(cell.strip() == '' for cell in row):
                    continue  # Skip row if all cells are empty
                elif any(cell.strip() == '' for cell in row):
                    raise ValueError("Error: Found an empty cell in a row")  # Raise error if any single cell is empty
                else:
                    cleaned_rows.append(row)  # Append row if it has no empty cells

            # Write cleaned rows back to a new CSV file or update the existing one
            with open(exp_results, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(actual_headers)
                csv_writer.writerows(cleaned_rows)

        except Exception as e:
            print("CSV format test failed:", str(e))
            return False

        for i, expected_row in enumerate(expected_headers):
            if i >= len(actual_headers):
                print("# CSV file doesn't have enough rows")
                return False
            if actual_headers[i] != expected_row:
                print("Rows don't match the expected values")
                return False
            return False

    if not mech.endswith('.cti'):
        print("mech format file failed:")
        return False

    return True

