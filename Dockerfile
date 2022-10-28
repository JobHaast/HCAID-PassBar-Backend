FROM python:3.10

RUN mkdir /app

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN conda install --file requirements.txt

CMD ["python", "api.py"]