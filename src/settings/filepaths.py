from pathlib import Path

src_dir = Path(__file__).parent.parent.as_posix()
project_dir = Path(src_dir).parent.as_posix()
mech_path = f"{project_dir}/resources/mech"
output_dir = f"{project_dir}/resources/output"
input_dir = f"{project_dir}/resources/input"