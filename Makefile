.PHONY: install run build up down logs clean

## Install Python dependencies into the active virtual environment
install:
	pip install -r requirements.txt

## Run the app locally (no Docker)
run:
	streamlit run app.py

## Build the Docker image
build:
	docker compose build

## Start the container in the background
up:
	docker compose up -d

## Stop the container
down:
	docker compose down

## Stream container logs
logs:
	docker compose logs -f

## Remove containers + the hf-cache volume (saved_index bind-mount is NOT deleted)
clean:
	docker compose down -v
