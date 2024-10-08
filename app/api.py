"""
    Websocket test
        wscat -H "Authorization: Bearer cwivox2023" -c ws://localhost:8501/chatbot
"""

import os
from fastapi import FastAPI, WebSocket
from langchain.agents import AgentExecutor
from chain_setup_llama import setup_agent
from langchain.input import get_colored_text
from datetime import datetime
import websockets
from agent_logger import AgentCallbackHandler
from starlette.websockets import WebSocketDisconnect

import logging

# Set up a logger
logger = logging.getLogger(__name__)

sender_info_dict = {
    'Human': 'Human request as an input',
    'AI': 'AI agent response as an output',
    'ToolUnity': 'requested name of a Unity tool',
    'Unity': 'response of a called Unity tool'
}

app = FastAPI()
agent_executor: AgentExecutor = setup_agent()

# Use a global variable to store the WebSocket connection
global_websocket = None
websocket_endpoint_chatbot_url = f"ws://{os.getenv('HOST')}:{int(os.getenv('PORT'))}/chatbot"
websocket_endpoint_chatbot_header = [("Authorization", f"Bearer {os.getenv('AUTHORIZER')}")]


async def authenticate_websocket(websocket: WebSocket):
    auth_header = websocket.headers.get("Authorization")
    if not auth_header or auth_header != f"Bearer {os.getenv('AUTHORIZER')}":
        await websocket.close(code=4000, reason="Authentication failed")

async def unity_simulation_response(function_name, **params):
    """
    Simulates various Unity and AR interfaces using a response dictionary.
    """
    function_name = function_name.strip()
    unity_callable_functions = {
        "StartAssemble": "Assembly process initiated successfully.",
        "NextStep": "Moved to the next assembly step.",
        "FrontStep": "Went back to the previous assembly step.",
        "Explode": "Triggered explosion for detailed viewing.",
        "Recover": "Restored the initial state of AR objects after an explosion.",
        "FinishedVideo": "Ended the assembly process. Showing a video of the assembled LEGO bricks.",
        "ReShow": "Repeating the current assembly step.",
        "Enlarge": "Enlarged or zoomed out the current object.",
        "Shrink": "Shrunk or zoomed in on the current object.",
        "GoToStep": f"Moved to assembly step {params}.",
        "Rotate": "Rotated the current object to the specified direction.",
        "ShowPieces": "Showing all candidate LEGO pieces to be assembled.",
        "HighlightCorrectComponents": "Highlighted correct attachment points and components.",
        "GetCurrentStep": "Got the number of the current step.",
        "GetRemainingStep": "Got the number of the remaining steps.",
        "CheckStepStatusVR": "Checked if the current step in Unity is accomplished correctly or not.",
        "APICallObjectRecognitionAR": "Called VLM to identify LEGO pieces based on provided video streaming data from AR glasses. Highlighting recognized pieces in the AR environment.",
        "APICallCheckStepStatusAR": "Called VLM to determine if the current assembly step is completed correctly or not, using the provided video streaming data from AR glasses."
    }

    if function_name in unity_callable_functions:
        unity_response = unity_callable_functions[function_name]
    else:
        unity_response = f"Sorry, function {function_name} is not implemented yet."

    return unity_response


# Use this event to open the WebSocket connection when the server starts
# Ensure that the WebSocket server is started before the FastAPI application. The WebSocket server needs to be running and accepting connections before your FastAPI application tries to connect to it.
@app.on_event("startup")
async def on_startup():
    global global_websocket
    try:
        global_websocket = await websockets.connect(websocket_endpoint_chatbot_url, extra_headers=websocket_endpoint_chatbot_header)
        print("WebSocket connection opened")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"WebSocket connection failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during WebSocket connection: {e}")


# Use this event to close the WebSocket connection when the server stops
@app.on_event("shutdown")
async def on_shutdown():
    global global_websocket
    if global_websocket and global_websocket.open:
        await global_websocket.close()
        print("WebSocket connection closed")


@app.post("/ask")
async def generate_response(query: str) -> str:
    try:
        response = agent_executor.run(query, callbacks=[AgentCallbackHandler()])
        print(get_colored_text("Response: >>>", "green"))
        print(get_colored_text(response, "green"))
    except Exception as e:
        print(get_colored_text(f"Failed to process {query}", "red"))
        print(get_colored_text(f"Error: {e}", "red"))
        response = str(e)
    return response


@app.websocket("/chatbot")
async def websocket_endpoint_chatbot(websocket: WebSocket):
    try:
        #await authenticate_websocket(websocket)
        await websocket.accept()
        print(get_colored_text(f"{datetime.now()}\t WebSocket connection established", "green"))

        while True:
            try:
                # Receive message
                message = await websocket.receive_text()
                print(get_colored_text(f"{datetime.now()}\t Server received <<<: {message}", "blue"))

                # Perform dialogue processing here
                # You can call Unity tools or any other logic
                if message.startswith('ToolUnity: '):
                    info = message.split(':')[1]
                    prefix_sender = 'Unity'
                    response = await unity_simulation_response(info)  # Call Unity App!
                
                else:
                    prefix_sender = 'AI'
                    response = await generate_response(message)  # Call AI agent!

                # For now, let's echo back the message
                await websocket.send_text(f'{prefix_sender}: {response}')
                print(get_colored_text(f"{datetime.now()}\t Server ({prefix_sender}) sent >>>: {response}", "red"))

            except WebSocketDisconnect:
                print(get_colored_text(f"{datetime.now()}\t WebSocket disconnected", "yellow"))
                break

    except Exception as e:
        print(get_colored_text(f"{datetime.now()}\t An error occurred: {str(e)}", "red"))

    finally:
        # This block will be executed when the connection is closed, either normally or due to an error
        print(get_colored_text(f"{datetime.now()}\t WebSocket connection closed", "yellow"))
