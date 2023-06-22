import csv

def test_input_files(path_to_mechanism, path_to_input, type: str):
    with open(path_to_input, 'r') as file:
        csv_reader = csv.reader(file)

        # Test the format
        try:
        except Exception as e:
            print("CSV format test failed:", str(e))
            return False

        # Test the column headers
        expected_headers = ['column1', 'column2', 'column3']
        actual_headers = next(csv_reader)

        if actual_headers != expected_headers:
            print("Column header test failed. Expected:", expected_headers, "Actual:", actual_headers)
            return False

    if not path_to_mechanism.endswith('.cti'):
        print("mech format file failed:")
        return False

    return True
