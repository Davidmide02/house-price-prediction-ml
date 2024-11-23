FROM python:3.12.4-slim

# RUN pip install 

WORKDIR /app

COPY ["midrequirement.txt", "app.py","model_svm.bin", "./" ]

# install dependencies on the system vot
RUN pip install -r midrequirement.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD [ "app.py" ]