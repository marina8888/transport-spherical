from pathlib import Path

src_dir = Path(__file__).parent.parent.as_posix()
project_dir = Path(src_dir).parent.as_posix()
mech_dir = f"{project_dir}/resources/mech"
output_dir = f"{project_dir}/resources/output"
output_dir_numerical = f"{project_dir}/resources/output/numerical"
output_dir_graphs = f"{project_dir}/resources/output/graphs"
input_dir = f"{project_dir}/resources/input"