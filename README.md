## Introduction
We develop an Augmented Reality Training Assistant (ARTA) workflow based on [LangChain](https://github.com/langchain-ai/langchain) framework, enlighten by the state-of-the-art [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/). 
In this work, we aim to introduce Large Language Models (LLMs) as human-like "brain" to augment Argument Reality (AR) system with artificial Intelligence; 
and vice versa, we enable Argument Reality (AR) system to provide services by external API tools to augment general LLMs specifically for the ARTA use case, so that AR system can better plan which tool to use and when to use based on fully understanding users' need.
Therefore, there are three key phrase as follows:
- Phase 1: Develop the workflow using existing LLMs (such as `gpt-3.5-turbo-16k-0613` and `llama2-7b-chat`).
- Phase 2: Finetune LLMs (such as llama2) with large-scale public datasets and replace the core LLM in the workflow.
- Phase 3: Create and evaluate ARTA dataset by playing with assistant under the workflow [TODO].

<img src="workflow.png" alt="workflow.png" width="75%"/>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Code License Usage and License Notices**: This project utilizes certain checkpoints that are subject to their respective original license. Users must comply with all terms and conditions of this original license, including but not limited to the specific license for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2). This project does not impose any additional constraints beyond those stipulated in the original license. Furthermore, users are reminded to ensure that their use of the checkpoints is in compliance with all applicable laws and regulations.

## How to run

To download the repository, you can use the following command:
```shell
git clone https://gitlab.com/horizon-europe-voxreality/dialogue-system/arta.git
```

### 1. Run locally

```shell
export PORT=8000
export HOST=localhost
cd arta/app
python main.py
```

### 2. Utilizing the API-Enabled Docker Image
To download (pull) the Docker image to your system, you can use the following command:

```
docker pull voxreality/training_agent:latest
```

This command will get the conference agent image from the voxreality repository.

To run the container you can use the following command:
```
docker run --gpus all -p 8000:8000 voxreality/training_agent:latest
```

### 3.  Build Docker Image from Code
To run the container you can use the following command:
```shell
# copy the codes in your GPU server
scp -r arta jpei@sprout.huiskamer.private:/home/jpei/ARTA-delivery/
# Enter the root dir of arta, e.g., cd /home/jpei/ARTA-delivery/arta
docker build -t api .
docker run --gpus all -p 8000:8000 -it api
```

We develop api by websocket to interactive communication between user (trainee) and assistant (AI trainer).
```shell
wscat -H "Authorization: Bearer cwivox2023" -c ws://localhost:8000/chatbot
```

Below is an example of a conversation in the interactive mode:

<img src="example.png" alt="example.png" width="50%"/>


And you can check with full example at [full_example.txt](full_example.txt).

```shell input
QUERY=$(python -c 'import urllib.parse; print(urllib.parse.quote("Can you start to teach me how to assemble a LEGO car?"))')

curl -X 'POST' \
  'http://localhost:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer cwivox2023' \
  -d '{"query": "'"$QUERY"'"}'
```

Some useful docker commands:
```shell
# List out all docker images
docker ps
# Stop a docker image
docker stop <container_id>
# Remove a docker image
docker rm <container_id>
# Remove Old Docker Containers and Images
docker container prune
docker image prune -a
```
Check with GPU usage:
```shell
watch -n 1 nvidia-smi
```

## How to change finetuned model?
You can change the model by assigning the model id such as "Jiahuan/vox-finetune-llama-2-7b-chat", which are public available on HuggingFace.
```shell
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Jiahuan/vox-finetune-llama-2-7b-chat", device_map="auto", trust_remote_code=True, token=True)
model = AutoModelForCausalLM.from_pretrained("Jiahuan/vox-finetune-llama-2-7b-chat", device_map="auto", trust_remote_code=True, token=True, quantization_config=bnb_config)
```