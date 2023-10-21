FROM python:3.11-slim-bullseye
ENV PYTHONUNBUFFERED=1

# installing poetry
RUN apt update -y && \
    apt install -y --no-install-recommends ssh curl git bash rsync psmisc && \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 - && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# adding poetry to path
ENV PATH=/etc/poetry/bin:$PATH

# copying application files
COPY . .

# set up poetry environment
RUN poetry config virtualenvs.in-project true && poetry install
 
# Start application
CMD [ ".venv/bin/python3", "src/main.py" ]