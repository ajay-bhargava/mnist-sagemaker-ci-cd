###
# Base Stage
# Setting up the base image with Python and creating a non-root user.
###

# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM radixai/python-gpu:3.11-cuda11.8 AS base

# Remove docker-clean so we can keep the apt cache in Docker build cache.
RUN rm /etc/apt/apt.conf.d/docker-clean

# Create a non-root user and switch to it.
ARG UID=1000
ARG GID=$UID
RUN groupadd --gid $GID user && \
    useradd --create-home --gid $GID --uid $UID user --no-log-init && \
    chown user /opt/
USER user

# Create and activate a virtual environment.
ENV VIRTUAL_ENV /opt/mnist-sagemaker-ci-cd-env
ENV PATH $VIRTUAL_ENV/bin:$PATH
RUN python -m venv $VIRTUAL_ENV

# Set the working directory.
WORKDIR /workspaces/mnist-sagemaker-ci-cd/


###
# Poetry Stage
# Installing Poetry and runtime Python dependencies.
###

FROM base as poetry

USER root

# Install Poetry in a separate virtual environment.
ENV POETRY_VERSION 1.7.0
ENV POETRY_VIRTUAL_ENV /opt/poetry-env
RUN --mount=type=cache,target=/root/.cache/pip/ \
    python -m venv $POETRY_VIRTUAL_ENV && \
    $POETRY_VIRTUAL_ENV/bin/pip install poetry~=$POETRY_VERSION && \
    ln -s $POETRY_VIRTUAL_ENV/bin/poetry /usr/local/bin/poetry

# Install compilers for certain packages or platforms.
RUN --mount=type=cache,target=/var/cache/apt/ \
    --mount=type=cache,target=/var/lib/apt/ \
    apt-get update && \
    apt-get install --no-install-recommends --yes build-essential tree

USER user

# Install the runtime Python dependencies in the virtual environment.
COPY --chown=user:user poetry.lock* pyproject.toml /workspaces/mnist-sagemaker-ci-cd/
RUN mkdir -p /home/user/.cache/pypoetry/ && mkdir -p /home/user/.config/pypoetry/ && \
    mkdir -p src/mnist_sagemaker_ci_cd/ && touch src/mnist_sagemaker_ci_cd/__init__.py && touch README.md
RUN --mount=type=cache,uid=$UID,gid=$GID,target=/home/user/.cache/pypoetry/ \
    poetry install --only main --no-interaction


###
# Development Stage
# Installing development tools and configuring the user's shell.
###

FROM poetry as dev

USER root

# Install development tools.
RUN --mount=type=cache,target=/var/cache/apt/ \
    --mount=type=cache,target=/var/lib/apt/ \
    apt-get update && \
    apt-get install --no-install-recommends --yes curl git gnupg ssh sudo vim zsh && \
    sh -c "$(curl -fsSL https://starship.rs/install.sh)" -- "--yes" && \
    usermod --shell /usr/bin/zsh user && \
    echo 'user ALL=(root) NOPASSWD:ALL' > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user
USER user

# Install development Python dependencies in the virtual environment.
RUN --mount=type=cache,uid=$UID,gid=$GID,target=/home/user/.cache/pypoetry/ \
    poetry install --no-interaction

# Persist output generated during docker build for the dev container.
COPY --chown=user:user .pre-commit-config.yaml /workspaces/mnist-sagemaker-ci-cd/
RUN mkdir -p /opt/build/poetry/ && cp poetry.lock /opt/build/poetry/ && \
    git init && pre-commit install --install-hooks && \
    mkdir -p /opt/build/git/ && cp .git/hooks/commit-msg .git/hooks/pre-commit /opt/build/git/

# Configure the non-root user's shell.
ENV ANTIDOTE_VERSION 1.8.6
RUN git clone --branch v$ANTIDOTE_VERSION --depth=1 https://github.com/mattmc3/antidote.git ~/.antidote/ && \
    echo 'zsh-users/zsh-syntax-highlighting' >> ~/.zsh_plugins.txt && \
    echo 'zsh-users/zsh-autosuggestions' >> ~/.zsh_plugins.txt && \
    echo 'source ~/.antidote/antidote.zsh' >> ~/.zshrc && \
    echo 'antidote load' >> ~/.zshrc && \
    echo 'eval "$(starship init zsh)"' >> ~/.zshrc && \
    echo 'HISTFILE=~/.history/.zsh_history' >> ~/.zshrc && \
    echo 'HISTSIZE=1000' >> ~/.zshrc && \
    echo 'SAVEHIST=1000' >> ~/.zshrc && \
    echo 'setopt share_history' >> ~/.zshrc && \
    echo 'bindkey "^[[A" history-beginning-search-backward' >> ~/.zshrc && \
    echo 'bindkey "^[[B" history-beginning-search-forward' >> ~/.zshrc && \
    mkdir ~/.history/ && \
    zsh -c 'source ~/.zshrc'


###
# Application Stage
# Preparing the application for deployment on App Runner
###

FROM base AS app

# Copy the virtual environment from the poetry stage.
COPY --from=poetry $VIRTUAL_ENV $VIRTUAL_ENV

# Copy the package source code to the working directory.
COPY --chown=user:user . .

# Expose the application.
ENTRYPOINT ["/opt/mnist-sagemaker-ci-cd-env/bin/poe"]
CMD ["api"]


###
# Sagemaker Stage
# Preparing the application for deployment on Sagemaker Training.
###

ARG PYTHON_VERSION=3.11
FROM radixai/python-gpu:3.11-cuda11.8 AS sagemaker-base

# Remove docker-clean so we can keep the apt cache in Docker build cache.
RUN rm /etc/apt/apt.conf.d/docker-clean

# Create a non-root user and switch to it.
ARG UID=1000
ARG GID=$UID
RUN groupadd --gid $GID user && \
    useradd --create-home --gid $GID --uid $UID user --no-log-init && \
    chown user /opt/
USER user

# Create and activate a virtual environment.
ENV VIRTUAL_ENV /opt/poetry/
ENV PATH $VIRTUAL_ENV/bin:$PATH
RUN python -m venv $VIRTUAL_ENV

# Set the working directory.
WORKDIR /opt/ml/

FROM sagemaker-base as sagemaker

USER root

# Install Poetry in a separate virtual environment.
ENV POETRY_VERSION 1.7.0
ENV POETRY_VIRTUAL_ENV /opt/poetry/
RUN --mount=type=cache,target=/root/.cache/pip/ \
    python -m venv $POETRY_VIRTUAL_ENV && \
    $POETRY_VIRTUAL_ENV/bin/pip install poetry~=$POETRY_VERSION && \
    ln -s $POETRY_VIRTUAL_ENV/bin/poetry /usr/local/bin/poetry

# Install compilers for certain packages or platforms.
RUN --mount=type=cache,target=/var/cache/apt/ \
    --mount=type=cache,target=/var/lib/apt/ \
    apt-get update && \
    apt-get install --no-install-recommends --yes build-essential tree

# Install the runtime Python dependencies in the virtual environment.
COPY --chown=user:user poetry.lock* pyproject.toml /opt/ml/
RUN mkdir -p /opt/.cache/pypoetry/
RUN --mount=type=cache,uid=$UID,gid=$GID,target=/opt/.cache/pypoetry/ \
    poetry install --only main,training --no-interaction

USER user

# Copy the package source code to the working directory.
COPY --chown=user:user src/mnist_sagemaker_ci_cd/ /opt/ml/code/

# Define Sagemaker Pytorch container environment variables to determine the user code directory
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# This is where SageMaker will look to find the entry script for training
ENV SAGEMAKER_PROGRAM train.py