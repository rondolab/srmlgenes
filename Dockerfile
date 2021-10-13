FROM continuumio/miniconda3

WORKDIR /app
ADD *.py  ./
ADD assets ./
ADD environment.yml ./

RUN conda env create -f environment.yml

ENTRYPOINT ["conda", "run", "-n", "srmlgenes"]
CMD ["gunicorn", "index:application"]