FROM python:3.9
WORKDIR /backend
COPY . /backend
COPY ../scripts/processing/model /backend
RUN pip3 install --upgrade pip -r requirements.txt
EXPOSE 5000
CMD python ./app.py