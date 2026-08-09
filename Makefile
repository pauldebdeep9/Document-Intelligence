.PHONY: install update lock env-check corpus ingest parse extract index ask eval serve test acl lint schemas slice

CONDA_ENV = Sai2608

install:
	conda env create -f environment.yml || echo "Env already exists? Run 'make update' instead."

update:
	conda env update -f environment.yml --prune

lock:
	conda env export --no-builds > environment.lock.yml

env-check:
	python -V
	which python
	conda info --envs

corpus:
	python scripts/gen_corpus.py --n 20 --types purchase_order --out data/synthetic
	python scripts/gen_acl_graph.py --out data/acl
	python scripts/gen_gold.py --corpus data/synthetic --out data/gold

ingest:  ; isc ingest --source data/synthetic
parse:   ; isc parse
extract: ; isc extract --doc-type purchase_order
index:   ; isc index
ask:     ; isc ask "$(Q)" --as $(USER_ID)
eval:    ; isc eval --harness both

# Walking skeleton, end to end, one command.
slice: corpus ingest parse extract index eval

serve:   ; uvicorn isc.api.app:app --reload --port 8000

test:    ; pytest -q
acl:     ; pytest -q -m acl          # never allowed to fail
lint:    ; ruff check src tests && mypy
schemas: ; python scripts/export_schemas.py --out schemas
