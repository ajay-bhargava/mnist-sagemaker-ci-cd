##########################
#   ____                 
#  |  _ \                
#  | |_) | __ _ ___  ___ 
#  |  _ < / _` / __|/ _ \
#  | |_) | (_| \__ \  __/
#  |____/ \__,_|___/\___|
#                                      
##########################

# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:$PYTHON_VERSION-slim AS base

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

####################################
#   _____           _              
#  |  __ \         | |             
#  | |__) |__   ___| |_ _ __ _   _ 
#  |  ___/ _ \ / _ \ __| '__| | | |
#  | |  | (_) |  __/ |_| |  | |_| |
#  |_|   \___/ \___|\__|_|   \__, |
#                             __/ |
#                            |___/ 
####################################

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

################################################################
#  __      _______  _____          _        _____             
#  \ \    / / ____|/ ____|        | |      |  __ \            
#   \ \  / / (___ | |     ___   __| | ___  | |  | | _____   __
#    \ \/ / \___ \| |    / _ \ / _` |/ _ \ | |  | |/ _ \ \ / /
#     \  /  ____) | |___| (_) | (_| |  __/ | |__| |  __/\ V / 
#    __\/_ |_____/ \_____\___/ \__,_|\___| |_____/ \___| \_/  
#   / ____|          | |      (_)                             
#  | |     ___  _ __ | |_ __ _ _ _ __   ___ _ __              
#  | |    / _ \| '_ \| __/ _` | | '_ \ / _ \ '__|             
#  | |___| (_) | | | | || (_| | | | | |  __/ |                
#   \_____\___/|_| |_|\__\__,_|_|_| |_|\___|_|      
#################################################################       

FROM poetry as dev

USER root

RUN --mount=type=cache,target=/var/cache/apt/ \
    --mount=type=cache,target=/var/lib/apt/ \
    apt-get update && \
    apt-get install --no-install-recommends --yes curl git gnupg ssh sudo vim zsh awscli nodejs npm && \
    sh -c "$(curl -fsSL https://starship.rs/install.sh)" -- "--yes" && \
    npm install serverless -g && \
    usermod --shell /usr/bin/zsh user && \
    echo 'user ALL=(root) NOPASSWD:ALL' > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user

USER user

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

############################################
#   ______        _            _____ _____ 
#  |  ____|      | |     /\   |  __ \_   _|
#  | |__ __ _ ___| |_   /  \  | |__) || |  
#  |  __/ _` / __| __| / /\ \ |  ___/ | |  
#  | | | (_| \__ \ |_ / ____ \| |    _| |_ 
#  |_|  \__,_|___/\__/_/    \_\_|   |_____|
#
############################################                                        
                                         

FROM base AS app

# Copy the virtual environment from the poetry stage.
COPY --from=poetry $VIRTUAL_ENV $VIRTUAL_ENV

# Copy the package source code to the working directory.
COPY --chown=user:user . .

# Expose the application.
ENTRYPOINT ["/opt/mnist-sagemaker-ci-cd-env/bin/poe"]
CMD ["api"]

################################################################
#    _____                                  _             
#   / ____|                                | |            
#  | (___   __ _  __ _  ___ _ __ ___   __ _| | _____ _ __ 
#   \___ \ / _` |/ _` |/ _ \ '_ ` _ \ / _` | |/ / _ \ '__|
#   ____) | (_| | (_| |  __/ | | | | | (_| |   <  __/ |   
#  |_____/ \__,_|\__, |\___|_| |_| |_|\__,_|_|\_\___|_|   
#                 __/ |                                   
#                |___/                                    
#
################################################################

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0-gpu-py310 as sagemaker
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wandb dvc"[s3]" && \
    pip freeze
ARG KEY_ID=""
ARG SECRET_KEY=""
ENV AWS_ACCESS_KEY_ID=${KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${SECRET_KEY}