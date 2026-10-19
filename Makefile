.PHONY: install models api ui test eval docker

install:          # run inside an activated virtual environment
	pip install -r requirements.txt

models:           # one-time download of the local search models (needs internet once)
	python scripts/download_models.py

api:
	uvicorn app.api.main:app --reload --port 8000

ui:
	streamlit run ui/streamlit_app.py

test:
	pytest -q

eval:             # make eval DOCS=path/to/docs DATASET=path/to/questions.jsonl
	python -m evaluation.run_eval --docs $(DOCS) --dataset $(DATASET)

docker:
	docker compose up --build
