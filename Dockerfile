# pull python base image
FROM python:3.10

# specify working directory
WORKDIR /survival_pred_api

ADD /requirements.txt .
ADD /xgboost-model.pkl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# copy application files
ADD /app/* ./app/

# expose port for application
EXPOSE 8080

# start fastapi application
CMD ["python", "app/main.py"]