FROM ubuntu 
RUN apt-get update 
RUN apt-get install -y python3 python3-pip
RUN pip install --upgrade pip
RUN pip3 install flask

RUN mkdir /opt/app

WORKDIR /opt/app
COPY . /opt/app

RUN pip install -r requirements.txt

ENTRYPOINT FLASK_APP=/opt/app/clientApp.py flask run --host=0.0.0.0