FROM continuumio/miniconda3

WORKDIR /app
ADD *.py  /app
ADD assets /app
ADD environment.yml /app

RUN conda env create -f environment.yml
RUN conda activate srmlgenes

CMD gunicorn index