from pathlib import Path
p=Path('src/dropout-julia/study_fixture_oracle.py')
source=p.read_text().split('sizes=')[0]
exec(compile(source,'oracle_function_definitions','exec'),{})
