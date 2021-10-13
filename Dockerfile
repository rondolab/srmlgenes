FROM continuumio/miniconda3

RUN conda create -f environment.yml
RUN conda activate srmlgenes

CMD gunicorn index