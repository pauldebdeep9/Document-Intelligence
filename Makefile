.PHONY: install update lock env-check corpus corpus-clean corpus-verify ingest parse extract index ask eval serve test acl lint schemas slice slice-clean smoke

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

# RUN_ID is empty by default (each stage mints its own via new_run_id() --
# fine standalone). `slice` below overrides it per sub-make invocation so
# all five stages share ONE id and runs/<id>/ holds the whole pipeline's
# trace together, not five scattered directories.
RUN_ID ?=
RUN_ID_FLAG = $(if $(RUN_ID),--run-id $(RUN_ID),)

ingest:  ; isc ingest --source data/synthetic $(RUN_ID_FLAG)
parse:   ; isc parse $(RUN_ID_FLAG)
extract: ; isc extract --doc-type purchase_order $(RUN_ID_FLAG)
index:   ; isc index $(RUN_ID_FLAG)
ask:     ; isc ask "$(Q)" --as $(USER_ID)
eval:    ; isc eval --harness both $(RUN_ID_FLAG)

# Everything ingest/parse/extract/index build that corpus-clean does not
# already cover -- the derived index state, which would otherwise leave
# orphaned chunks/documents from a previous slice lying around even after
# the corpus itself is regenerated.
slice-clean: corpus-clean
	rm -f data/docstore.sqlite data/vector_store.pkl
	rm -rf data/blobs

# Walking skeleton, end to end, one command, from clean. One run id shared
# by every isc stage (see RUN_ID_FLAG above) so runs/<id>/ holds the
# complete trace, all per-stage artifacts, and both eval reports together.
# `&&` between stages: a non-zero exit stops the chain rather than silently
# continuing into a later stage built on incomplete upstream state.
slice: slice-clean
	@RUN_ID=$$(python -c "from isc.common.ids import new_run_id; print(new_run_id())"); \
	echo "run id: $$RUN_ID"; \
	$(MAKE) corpus && \
	$(MAKE) ingest RUN_ID=$$RUN_ID && \
	$(MAKE) parse RUN_ID=$$RUN_ID && \
	$(MAKE) extract RUN_ID=$$RUN_ID && \
	$(MAKE) index RUN_ID=$$RUN_ID && \
	$(MAKE) eval RUN_ID=$$RUN_ID

serve:   ; uvicorn isc.api.app:app --reload --port 8000

test:    ; pytest -q
acl:     ; pytest -q -m acl          # never allowed to fail
lint:    ; ruff check src tests && mypy
schemas: ; python scripts/export_schemas.py --out schemas
smoke:   ; python scripts/smoke_llm.py
