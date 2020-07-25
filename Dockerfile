FROM python:3.7-slim

EXPOSE 8501

RUN apt-get update -y \
    && apt-get install -y gcc \
    
    #python-setuptools \
    #python3-pip \
    && apt-get clean \
    && apt-get autoremove


# Install pip requirements

ADD requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
ADD . /app


ENTRYPOINT [ "streamlit", "run"]
CMD ["app.py"]