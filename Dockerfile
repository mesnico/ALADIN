FROM continuumio/miniconda3

WORKDIR /app
ARG WORKSPACE=/app
ARG DATA=/data

ENV PYTHONPATH=/app

ARG USER=runner
ARG GROUP=runner-group
# ARG SRC_DIR=src

# Create non-privileged user to run
# RUN addgroup --system ${GROUP} && \
#    adduser --system --ingroup ${GROUP} ${USER} && \
#    chown -R ${USER}:${GROUP} ${WORKSPACE}

# Change to non-privileged user
# USER ${USER}

# upgrade conda
RUN conda update -n base -c defaults conda

# Copy the all the stuff to the current workspace
COPY . ${WORKSPACE}/

# Create the environment and install dependencies
RUN conda env create -f ${WORKSPACE}/environment.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "oscar", "/bin/bash", "-c"]

# Make sure the environment is activated:
# RUN echo "Make sure flask is installed:"
# RUN python -c "import flask"

# Expose port 5020
EXPOSE 5020

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "oscar", "python", "alad/extraction/vbs/query_server.py"]

