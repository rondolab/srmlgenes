FROM mambaorg/micromamba

COPY --chown=micromamba:micromamba environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

WORKDIR /app
ADD *.py  ./
ADD assets ./

CMD ["gunicorn", "index:application"]