FROM python:3.9.6

COPY requirements.txt /
RUN python3 -m pip install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000
CMD [ "python" , "app.py"]