FROM continuumio/miniconda3

RUN conda env create -f environment.yml
RUN conda activate srmlgenes

CMD gunicorn index