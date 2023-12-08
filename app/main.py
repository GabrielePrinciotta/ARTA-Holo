"""
    This cli script to play with training assistant agent in the command line.
"""
import langchain
from langchain.input import get_colored_text
from langchain.agents import AgentExecutor
from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.history import FileHistory

from agent_logger import AgentCallbackHandler
from chain_setup_llama import setup_agent
# langchain.debug = True


def interactive():

    agent_executor: AgentExecutor = setup_agent()

    session = PromptSession(history=FileHistory(".agent-history-file"))
    while True:
        query = session.prompt(
            HTML("<b>Type <u>Your query</u></b>  ('q' to exit): ")
        )
        if query.lower() == 'q':
            break
        if len(query) == 0:
            continue
        try:
            print(get_colored_text("Response: >>> ", "green"))
            print(get_colored_text(agent_executor.run(query, callbacks=[AgentCallbackHandler()]), "green"))
        except Exception as e:
            print(get_colored_text(f"Failed to process {query}", "red"))
            print(get_colored_text(f"Error {e}", "red"))


if __name__ == "__main__":
    interactive()

