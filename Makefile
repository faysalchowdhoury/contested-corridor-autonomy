PY ?= python

.PHONY: setup data test experiments figures clean
setup:
	$(PIP) install -r requirements.txt || pip install -r requirements.txt

data:            ## ingest real data + build the corridor tensor
	$(PY) -c "import sys; sys.path.insert(0,'.'); from pathlib import Path; \
	from src.data.ingest import Manifest; from src.data.align import build_corridor; \
	from src.utils.config import load_configs; cfg=load_configs('configs')['data']; \
	m=Manifest(Path('data/manifest.json')); \
	build_corridor(cfg, Path('data/raw'), manifest=m, processed_dir=Path('data/processed')); m.write()"

test:
	$(PY) -m pytest -q

experiments:     ## full benchmark -> results/tables + results/figures
	$(PY) experiments/run_all_experiments.py

experiments-fast:
	$(PY) experiments/run_all_experiments.py --fast

figures: experiments

clean:
	rm -rf .pytest_cache results/tables/* results/figures/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
