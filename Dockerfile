# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /sports-betting

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["flask", "run"]
