.PHONY: install test lint run docker-build docker-run

install:
	pip install -r requirements.txt

test:
	pytest test_app.py -v

lint:
	flake8 app.py train.py test_app.py

run:
	flask run

docker-build:
	docker build -t sentiment-nlp-app .

docker-run:
	docker run -p 5000:5000 sentiment-nlp-app
