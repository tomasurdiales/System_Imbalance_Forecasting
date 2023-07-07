FROM python:3.10

# Declare environment variables
ENV PATH="/root/.local/bin:$PATH"

# Install Poetry
RUN apt-get -qq update && apt-get -qq -y install curl

# Set the working directory \
WORKDIR /app

# Install dependencies
RUN pip3 install pre-commit