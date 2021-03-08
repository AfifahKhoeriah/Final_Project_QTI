FROM python 3.6.8

WORKDIR /app 
COPY . .

ADD . /app
RUN pip install -r requirements.txt

ENTRYPOINT["python"]

CMD["main.py"]