from pathlib import Path

src_dir = Path(__file__).parent.parent.as_posix()
project_dir = Path(src_dir).parent.as_posix()
mech_dir = f"{project_dir}/resources/mech-yaml"
output_dir = f"{project_dir}/resources/output"
output_dir_numerical_output = f"{project_dir}/resources/output/numerical_output"
output_dir_numerical_domain = f"{project_dir}/resources/output/numerical_domain"
output_dir_graphs = f"{project_dir}/resources/output/graphs"
input_dir = f"{project_dir}/resources/input"