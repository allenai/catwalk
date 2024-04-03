# This Dockerfile describes a common base image for catwalk experiments in Beaker.

FROM ghcr.io/allenai/pytorch:1.12.0-cuda11.3-python3.9

# Initialize conda for bash.
# See https://stackoverflow.com/a/58081608/4151392
RUN eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Install GitHub CLI.
RUN conda install gh --channel conda-forge

RUN conda run -n base pip install --upgrade pip

RUN pip install Cython==0.29.36
RUN pip install spacy==3.0.6 --no-build-isolation

COPY requirements.txt requirements.txt

RUN conda run -n base pip install -r requirements.txt

RUN rm requirements.txt
