FROM python:3.11

# Set environment variables
# Credential for websocket communication
ENV AUTHORIZER="cwivox2023"
ENV HOST=localhost
ENV PORT=8000
ENV DEBUG=True
ENV HF_HOME=~/.cache/huggingface
# Credential for huggingface hub
ENV HF_HOME_TOKEN="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"

# Set the working directory
WORKDIR /app

# Install Git
RUN apt-get update && apt-get install -y git

# Copy the requirements file
COPY ./requirements.txt /app/
#COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Login huggingface
RUN huggingface-cli login --token $HF_HOME_TOKEN

# Copy the application code
#COPY . .
COPY ./app /app/

# Expose the port that your FastAPI application will run on
EXPOSE 8000

# Command to run the application
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000"]