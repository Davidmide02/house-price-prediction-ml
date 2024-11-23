FROM python:3.8.12-slim

RUN pip install 

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", ]

# install dependencies on the system vot

RUN pipenv install --system --deploy
