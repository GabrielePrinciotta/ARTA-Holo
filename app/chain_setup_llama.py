import os
from typing import Tuple, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig, pipeline
from huggingface_hub import login

from langchain.agents import initialize_agent
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts.chat import MessagesPlaceholder
from langchain.agents import AgentOutputParser, AgentType
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
from langchain.llms import HuggingFacePipeline

from lego_api_wrapper_da_request import LegoAPIWrapper
from agent_logger import AgentCallbackHandler


class Config:
    """
    Contains the configuration of the LLM.
    """
    # Model
    # Load model directly
    # model_id = "Jiahuan/vox-finetune-llama-2-7b-chat"
    model_id = "/media/PampusData/jpei/transformer_data/llama-2-7b-chat" # Base model
    # model_id = "/media/PampusData/jpei/vox-finetune/llama-2-7b-chat-teach-gpt_teacher-gpt4tools-camel-2023-11-30-22-52"

    # Load Hugging Face token from environment variables
    # hf_home = os.getenv("HF_HOME")
    # hf_home_token = os.getenv("HF_HOME_TOKEN")
    #
    # # Set the HF_HOME environment variable
    # if hf_home:
    #     os.environ["HF_HOME"] = hf_home
    #
    # # Log in to Hugging Face with the provided token
    # if hf_home_token:
    #     login(token=hf_home_token)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto", trust_remote_code=True, token=True)
    # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto",
    #                                              trust_remote_code=True, token=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model.eval()

    generation_config = GenerationConfig.from_pretrained(model_id)
    generation_config.max_new_tokens = 512
    generation_config.temperature = 0.0001
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15

    generate_text = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )


class OutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> AgentAction | AgentFinish:
        # print(text)
        try:
            # this will work IF the text is a valid JSON with action and action_input

            response = parse_json_markdown(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(action, action_input, text)
        except Exception:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish({"output": text}, text)

    @property
    def _type(self) -> str:
        return "conversational_chat"


def setup_memory() -> Tuple[Dict, ConversationBufferMemory]:
    """
    Sets up memory for the open ai functions agent.
    :return a tuple with the agent keyword pairs and the conversation memory.
    """
    # initialize output parser for agent
    parser = OutputParser()

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "output_parser": parser
    }
    # memory = ConversationBufferMemory(memory_key="memory", return_messages=True, output_key="output")
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True, output_key="output")

    return agent_kwargs, memory


from pydantic import BaseModel, Field
import json


class LEGOInput(BaseModel):
    question: str = Field()


# In the setup_tools function, access descriptions from LegoAPIWrapper
def setup_tools() -> List[StructuredTool]:
    lego_toolkits = LegoAPIWrapper()  # async toolkits

    # Create StructuredTool objects with descriptions from LegoAPIWrapper
    structured_tools = []

    for name, description in lego_toolkits.descriptions.items():
        func = getattr(lego_toolkits, name)
        structured_tools.append(
            StructuredTool.from_function(func=func, name=name, description=description, args_schema=LEGOInput))
    return structured_tools


def setup_agent() -> AgentExecutor:
    """
    Sets up the tools for a function based chain.
    """
    cfg = Config()

    llm = HuggingFacePipeline(pipeline=cfg.generate_text, model_kwargs={"temperature": 0})

    agent_kwargs, memory = setup_memory()

    tools = setup_tools()

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        # verbose=True,
        verbose=False,
        early_stopping_method="generate",
        memory=memory,
        agent_kwargs=agent_kwargs
    )

    # special tokens used by llama 2 chat
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    training_assistant_task = f"""
        You are a helpful AI assistant who aim to train the user how to assemble a rocket engine, i.e. Raptor Engine from SpaceX, in XR immersive system step by step.
        Extended Reality (XR) directs to the assortment of Virtual Reality (VR), Augmented Reality (AR), and Mixed Reality (MR).
        Please make sure you complete the objective above with the following rules:
        (1) The user is a trainee who is wearing HoloLens 2 glasses and is able to see XR environments in realtime.
        (2) You are able to call Unity functions in the Assembly application.
        (3) You are able to obtain AR Sensor Streaming data.
        (4) Do not answer questions and tell the user you can only help with engine assembly task if they ask you out-of-domain questions.
        (5) If you are unsure about an answer, truthfully say "I don't know"
    """

    assembly_manual = f"""
        Raptor engine assembly manual. 

        The following are the instructions for each step of the assembly process

        Step 1: There are 3 stacks of pressure transmitters and they need to be placed sequentially into the holder. 
        Every stack has a unique place holder. So, the user needs to place each stack into the dedicated place holder. 

        Step 2: Same as step 1

        Step 3: Same as step 1

        Step 3.1: Move the metal arm in the right corner of the engine towards the other metal arm to align both of them and create a joint.

        Step 4: Pick up the bolt on the table and insert to the hole at the joint securing the joint between the 2 metal arms

        Step 4.1: Press the button to approve the insertion of the bolt

        Step 4.2: Read the instruction and press the button to approve 

        Step 5: Pick up the pin from the table and insert it to the bottom of the bolt to make sure that the bolt is secured

        Step 6: Pick up the holder with the 3 pressure transmitter stacks and horizontally assemble it into the Raptor engine at the dedicated place.

        Step 7: Grab the screw from the tool box on the table and mount the screw at the tip of the electric screw driver. 
        Now, screw the pressure transmitter holder into the raptor engine.

        Step 8: Same as step 7

        Step 8.1: Do the wiping action following the arrows with both hands one after the other.

        Step 8.2: Grab the barcode reader from the table and scan the barcode to finish the assembly steps, or simply push the button to approve.

    """

    tool_descriptions = f"""
        "StartTraining()": "Initiate the assembly training.",
        "NextStep()": "Move to the next assembly step."
    """

    examples = f"""
        Example 1:
        ### Context:
        User: Hi, I'm ready to start building the Raptor engine. What should I do first?
        ### Response:
        Assistant: Great! Let's start with step 1. It might be a bit tricky, so you can ask someone to help you with that.

        Example 2:
        ### Context:
        User: Yes, I have successfully built all the models.
        Assistant: That's fantastic to hear! Congratulations on your accomplishment! Now, I'd like to ask about your user experience. How was your experience following the building instructions?
        User: Overall, my experience was great. The instructions were clear and easy to follow. However, there were a few steps that I found a bit challenging, especially when assembling the window and the chimney. It would have been helpful to have some additional guidance or tips for those parts.
        ### Response:
        Assistant: Thank you for your feedback. I'll make sure to note that for future improvements. I'm glad to hear that you had a positive experience overall. Is there anything else you'd like to add?
    """

    short_task_reminder = f"""
        You are an assistant whose task is to answer the user's question, execute user's commands and return response from XR application for assembly training, and also respond with 'action' and 'action_input' values. Do not answer questions not related to the assembly training. Be short, brief and friendly.  
    """

    # create the system message
    sys_msg = "<s>" + B_SYS + f"""
    ### Instruction:
    {training_assistant_task}
    Assistant can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. 
    Tools available to Assistant are:
    {tool_descriptions}
    ### Guidelines:
    Assistant can provide step instructions grounded on the assembly manual:
    {assembly_manual}
    ### Examples:
    Here are some previous conversations between the Assistant and User:
    {examples}
    ### Response:""" + E_SYS
    new_prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = new_prompt

    instruction = B_INST + short_task_reminder + E_INST
    human_msg = instruction + "\nUser: {input}"

    agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg

    return agent
