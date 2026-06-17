.PHONY: install run test docker
install:
	pip install -r requirements.txt
run:
	python main.py
test:
	pytest -q
docker:
	docker build -t health-risk-prediction . && docker run --rm health-risk-prediction
