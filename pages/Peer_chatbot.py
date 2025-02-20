import os
import logging
import queue
import json
import time

import streamlit as st
from openai import OpenAI
from streamlit import session_state as ss
from dotenv import load_dotenv

load_dotenv()

st.cache_resource(show_spinner=False)

def init_logging():
    logging.basicConfig(format="[%(asctime)s] %(levelname)+8s: %(message)s")
    local_logger = logging.getLogger()
    local_logger.setLevel(logging.INFO)
    return local_logger
logger = init_logging()

st.cache_resource(show_spinner=False)
def create_assistants_client():
    logger.info("Creating OpenAI client")
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return openai_client
client: OpenAI = create_assistants_client()

def add_message_to_state_session(message):
    if len(message) > 0:
        ss.messages.append({"role": "assistant", "content": message})
        
        
if 'tool_requests' not in ss:
    ss['tool_requests'] = queue.Queue()
tool_requests = ss['tool_requests']

def hello_world(name: str) -> str:
    time.sleep(5) # Demonstrate a long-running function
    return f"Hello {name}!"


def handle_requires_action(tool_request):
    st.toast("Running a function", icon=":material/function:")
    tool_outputs = []
    data = tool_request.data
    for tool in data.required_action.submit_tool_outputs.tool_calls:
        if tool.function.arguments:
            function_arguments = json.loads(tool.function.arguments)
        else:
            function_arguments = {}
        match tool.function.name:
            case "hello_world":
                logger.info("Calling hello_world function")
                answer = hello_world(**function_arguments)
                tool_outputs.append({"tool_call_id": tool.id, "output": answer})
            case _:
                logger.error(f"Unrecognized function name: {tool.function.name}. Tool: {tool}")
                ret_val = {
                    "status": "error",
                    "message": f"Function name is not recognize. Make sure you submit the request with the correct "
                               f"request structure. Fix your request and try again"
                }
                tool_outputs.append({"tool_call_id": tool.id, "output": json.dumps(ret_val)})
    st.toast("Function completed", icon=":material/function:")
    return tool_outputs, data.thread_id, data.id        
        
def data_streamer():
    """
    Stream data from the assistant. Text messages are yielded. Images and tool requests are put in the queue.
    :return:
    :rtype:
    """
    logger.info(f"Starting data streamer on {ss.stream}")
    st.toast("Thinking...", icon=":material/emoji_objects:")
    content_produced = False
    for response in ss.stream:
        match response.event:
            case "thread.message.delta":
                content = response.data.delta.content[0]
                match content.type:
                    case "text":
                        value = content.text.value
                        content_produced = True
                        yield value
                    case "image_file":
                        logger.info(f"Image file: {content}")
                        image_content = io.BytesIO(client.files.content(content.image_file.file_id).read())
                        content_produced = True
                        yield Image.open(image_content)
            case "thread.run.requires_action":
                logger.info(f"Run requires action: {response}")
                tool_requests.put(response)
                if not content_produced:
                    yield "[LLM requires a function call]"
                return
            case "thread.run.failed":
                logger.error(f"Run failed: {response}")
                return
    st.toast("Completed", icon=":material/emoji_objects:")
    logger.info(f"Finished data streamer on {ss.stream}")
        
def display_stream(content_stream, create_context=True):
    ss.stream = content_stream
    if create_context:
        with st.chat_message("assistant", avatar="👩‍🏭"):
            response = st.write_stream(data_streamer)
    else:
        response = st.write_stream(data_streamer)
    if response is not None:
        if isinstance(response, list):
            # Multiple messages in the response
            for message in response:
                add_message_to_state_session(message)
        else:
            # Single message in response
            add_message_to_state_session(response)        
        
        
def run():
    if "assistant" not in ss:
        assistant = client.beta.assistants.retrieve(assistant_id=os.environ["PEER_ID"])
        if assistant is None:
            raise RuntimeError(f"Assistant not found.")
        logger.info(f"Located assistant: {assistant.name}")
        ss["assistant"] = assistant
    assistant = ss["assistant"]

    st.set_page_config(page_title="Expert chatbot", layout="centered")
    st.title(f"Expert chatbot")

    if "messages" not in st.session_state:
        ss.messages = []

    # Display chat messages from state session on streamlit
    for message in ss.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask the EDA assistant"):
        # Display user message and add to history
        with st.chat_message("user"):
            st.write(prompt)
        ss.messages.append({"role": "user", "content": prompt})

        # Create a new thread if not already created
        if 'thread' in ss:
            thread = ss['thread']
        else:
            thread = client.beta.threads.create()
            logger.info(f"Created new thread: {thread.id}")
            ss['thread'] = thread

        # Add user message to the thread
        client.beta.threads.messages.create(thread_id=thread.id,
                                            role="user",
                                            content=prompt)

        # Create a new run with stream
        with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id
        ) as stream:
            # Start writing assistant messages to chat
            display_stream(stream)
            while not tool_requests.empty():
                logger.info("Handling tool requests")
                with st.chat_message("assistant", avatar="👩‍🏭"):
                    tool_outputs, thread_id, run_id = handle_requires_action(tool_requests.get())
                    with client.beta.threads.runs.submit_tool_outputs_stream(
                            thread_id=thread_id,
                            run_id=run_id,
                            tool_outputs=tool_outputs
                    ) as tool_stream:
                        display_stream(tool_stream, create_context=False)


run()