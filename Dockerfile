FROM python:3.9-bullseye
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt /code/
RUN apt-get update && apt-get install -y gettext
RUN pip install --no-cache-dir -r requirements.txt
