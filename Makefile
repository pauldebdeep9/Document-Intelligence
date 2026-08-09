.PHONY: install update lock env-check corpus corpus-clean corpus-verify ingest parse extract index ask eval serve test acl lint schemas slice smoke

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

# Order matters: masters before corpus (the corpus must be consistent with the
# master data for Signal.MASTER_DATA to mean anything), acl graph before the
# fidelity tests that check the adversarial pairs fire.
corpus:
	python scripts/gen_masters.py --out data/masters
	python scripts/gen_acl_graph.py --out data/acl
	python scripts/gen_corpus.py --n 20 --types purchase_order --out data/synthetic

corpus-clean:
	rm -f data/synthetic/*.pdf data/synthetic/*.acl.json data/synthetic/manifest.json
	rm -f data/gold/extraction/*.json

corpus-verify:
	pytest -q tests/integration/test_corpus_fidelity.py

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
smoke:   ; python scripts/smoke_llm.py
