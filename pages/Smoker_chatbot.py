import os
import logging
import queue
import json
import time
import requests
from io import StringIO
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
        ss.messages_smoker.append({"role": "assistant", "content": message})
        
def get_smoker_personality():
    """Read the smoker personality file content"""
    try:
        with open("llm_personalitites/smoker.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read personality file: {e}")
        return "You are a peer who is also trying to quit smoking, providing support and advice."

def call_infomaniak_llama(prompt, smoker_context, model_name="llama3"):
    """Call Infomaniak's Llama 3.3 API with the provided prompt and context"""
    try:
        product_id = os.getenv("INFOMANIAK_PRODUCT_ID", "")
        api_key = os.getenv("INFOMANIAK_API_KEY", "")
        
        logger.info(f"Calling Infomaniak Llama API with product ID: {product_id}")
        
        if not product_id or not api_key:
            logger.error("Infomaniak API credentials not found")
            st.error("Infomaniak API credentials not found. Please set INFOMANIAK_PRODUCT_ID and INFOMANIAK_API_KEY in .env file.")
            return None
            
        url = f"https://api.infomaniak.com/1/ai/{product_id}/openai/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Make sure we have the right history format for the API
        conversation_history = []
        if "messages_smoker" in ss and len(ss.messages_smoker) > 0:
            # Convert history to appropriate format - only include relevant context
            # Limit to last 10 messages to avoid token limits
            for msg in ss.messages_smoker[-10:]:
                conversation_history.append({"role": msg["role"], "content": msg["content"]})
        
        # Always add the system message first
        messages = [{"role": "system", "content": smoker_context}]
        # Add conversation history
        messages.extend(conversation_history)
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
            "max_tokens": 1024
        }
        
        response = requests.post(url, headers=headers, json=payload, stream=True)
        logger.info(f"Received response with status code: {response.status_code}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calling Infomaniak API: {e}")
        st.error(f"Error connecting to Infomaniak API: {str(e)}")
        return None

def stream_llama_response(response):
    """Stream the response from Llama API"""
    if response is None:
        logger.error("Llama API response is None")
        yield "Error connecting to Llama API"
        return "Error connecting to Llama API"
        
    buffer = StringIO()
    try:
        logger.info(f"Starting to stream Llama response, status code: {response.status_code}")
        
        if response.status_code != 200:
            error_msg = f"API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            yield error_msg
            return error_msg
            
        for line in response.iter_lines():
            if not line:
                continue
                
            line_str = line.decode('utf-8')
            
            # Check for data: prefix (SSE format)
            if line_str.startswith('data: '):
                try:
                    data = line_str[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        continue
                        
                    try:
                        # Try parsing as JSON
                        json_data = json.loads(data)
                        
                        # Handle OpenAI-compatible format
                        if 'choices' in json_data and json_data['choices']:
                            content = None
                            
                            # Try different possible formats
                            if 'delta' in json_data['choices'][0]:
                                content = json_data['choices'][0]['delta'].get('content')
                            elif 'text' in json_data['choices'][0]:
                                content = json_data['choices'][0].get('text')
                            elif 'message' in json_data['choices'][0]:
                                content = json_data['choices'][0]['message'].get('content')
                                
                            if content:
                                buffer.write(content)
                                yield content
                        
                    except json.JSONDecodeError:
                        # Not JSON, use as raw text
                        if data != '[DONE]':
                            buffer.write(data)
                            yield data
                        
                except Exception as e:
                    logger.error(f"Error processing line: {e}")
            else:
                # Try parsing as JSON directly
                try:
                    json_data = json.loads(line_str)
                    if 'choices' in json_data and json_data['choices']:
                        content = None
                        if 'delta' in json_data['choices'][0]:
                            content = json_data['choices'][0]['delta'].get('content')
                        elif 'text' in json_data['choices'][0]:
                            content = json_data['choices'][0].get('text')
                        elif 'message' in json_data['choices'][0]:
                            content = json_data['choices'][0]['message'].get('content')
                            
                        if content:
                            buffer.write(content)
                            yield content
                except json.JSONDecodeError:
                    # Not JSON, treat as raw text
                    buffer.write(line_str)
                    yield line_str
                    
    except Exception as e:
        logger.error(f"Error streaming Llama response: {e}")
        error_msg = f"Error: {str(e)}"
        yield error_msg
        return error_msg
    
    # Return the full text for history
    full_text = buffer.getvalue()
    buffer.close()
    return full_text

def test_infomaniak_api():
    """Test the Infomaniak API with a simple request, no streaming"""
    try:
        product_id = os.getenv("INFOMANIAK_PRODUCT_ID", "")
        api_key = os.getenv("INFOMANIAK_API_KEY", "")
        
        if not product_id or not api_key:
            return "Error: API credentials not found"
            
        url = f"https://api.infomaniak.com/1/ai/{product_id}/openai/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ],
            "temperature": 0.7,
            "stream": False
        }
        
        logger.info(f"Testing API with non-streaming request")
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            try:
                json_data = response.json()
                if 'choices' in json_data and json_data['choices']:
                    content = json_data['choices'][0]['message']['content']
                    return f"API test successful! Response: {content}"
            except Exception as e:
                return f"API returned 200 but could not parse response: {e}"
        else:
            return f"API test failed with status {response.status_code}"
            
    except Exception as e:
        return f"API test error: {e}"

# Keep existing tool_requests and related functions
if 'tool_requests' not in ss:
    ss['tool_requests'] = queue.Queue()
tool_requests = ss['tool_requests']

def handle_requires_action(tool_request):
    # Keep existing code
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
    # Keep existing code
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
        with st.chat_message("assistant", avatar="🐦"):
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
    # Set page config first
    st.set_page_config(page_title="Smoker Chatbot", layout="centered")
    
    # Assistant session state key
    if "smoker_assistant" not in ss:
        assistant = client.beta.assistants.retrieve(assistant_id=os.environ["SMOKER_ID"])
        if assistant is None:
            raise RuntimeError(f"Assistant not found.")
        logger.info(f"Located assistant: {assistant.name}")
        ss["smoker_assistant"] = assistant
    assistant = ss["smoker_assistant"]
    
    # Load personality
    smoker_context = get_smoker_personality()

    # Create columns for title and model selection
    title_col, model_col = st.columns([3, 1])
    
    with title_col:
        # Show title
        st.title("Ex-Smoker")
    
    with model_col:
        # Dropdown for model selection next to title
        if "selected_model_smoker" not in st.session_state:
            ss.selected_model_smoker = "OpenAI"
            
        selected_model = st.selectbox(
            "Model",
            ["OpenAI", "Llama 3.3"],
            key="model_selector_smoker",
            on_change=lambda: setattr(ss, "selected_model_smoker", ss.model_selector_smoker)
        )
    
    # Add API test button
    with st.sidebar:
        st.title("API Testing")
        if st.button("Test Infomaniak API"):
            with st.spinner("Testing API..."):
                result = test_infomaniak_api()
                st.text_area("API Test Result", result, height=200)

    # Initialize message history
    if "messages_smoker" not in st.session_state:
        ss.messages_smoker = []
        
    # Check if intro has been displayed
    if "smoker_intro_displayed" not in st.session_state:
        ss.smoker_intro_displayed = False

    # Display existing messages, skipping the hidden prompt
    for i, message in enumerate(ss.messages_smoker):
        # Skip only the hidden user prompt message
        if message["role"] == "user" and "Bonjour, parle-moi de ton expérience avec le tabac." in message["content"]:
            continue
            
        # Display all other messages, including the first assistant response
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="👤"):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Stream introduction message if it's the first load
    if not ss.smoker_intro_displayed:
        # Add hidden user message to history
        ss.messages_smoker.append({"role": "user", "content": "Bonjour, parle-moi de ton expérience avec le tabac."})
        
        # Stream the assistant's introduction message
        intro_message = "Salut! Moi c'est Mr Doe. Comme toi, j'essaie d'arrêter de fumer. J'ai commencé à 16 ans, et maintenant j'essaie de réduire ma consommation. Certains jours sont plus difficiles que d'autres... Je comprends totalement les défis qu'on rencontre quand on essaie de décrocher. Les envies soudaines, la nervosité, les habitudes ancrées... Mais on est ensemble dans cette bataille! Dis-moi, où en es-tu dans ton parcours avec la cigarette?"
        
        with st.chat_message("assistant", avatar="👤"):
            # Create a generator that yields parts of the intro message to simulate typing
            def stream_intro():
                words = intro_message.split()
                for i in range(0, len(words), 3):  # Send 3 words at a time for a smooth effect
                    chunk = " ".join(words[i:i+3]) + " "
                    yield chunk
                    time.sleep(0.1)  # Brief pause between chunks
                    
            # Stream the intro message
            st.write_stream(stream_intro)
        
        # Add intro to history after displaying
        ss.messages_smoker.append({"role": "assistant", "content": intro_message})
        
        # Mark intro as displayed
        ss.smoker_intro_displayed = True
    
    # Use standard Streamlit chat input
    prompt = st.chat_input("Discutez avec un ex-fumeur...")
    
    # Handle user input and generate responses
    if prompt:
        # Add user message to history and display it
        with st.chat_message("user"):
            st.write(prompt)
            
        ss.messages_smoker.append({"role": "user", "content": prompt})
        
        # Process based on the selected model
        if ss.model_selector_smoker == "OpenAI":
            # OpenAI processing
            if 'smoker_thread' in ss:
                thread = ss['smoker_thread']
            else:
                thread = client.beta.threads.create()
                logger.info(f"Created new thread: {thread.id}")
                ss['smoker_thread'] = thread

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
                    tool_outputs, thread_id, run_id = handle_requires_action(tool_requests.get())
                    with client.beta.threads.runs.submit_tool_outputs_stream(
                            thread_id=thread_id,
                            run_id=run_id,
                            tool_outputs=tool_outputs
                    ) as tool_stream:
                        with st.chat_message("assistant", avatar="👤"):
                            display_stream(tool_stream, create_context=False)
        else:
            # Llama 3.3 processing
            logger.info("Using Llama 3.3 model")
            
            # Test different model names
            model_names = ["llama-3.3-70b", "llama3"]
            
            response = None
            for model_name in model_names:
                try:
                    logger.info(f"Trying with model name: {model_name}")
                    response = call_infomaniak_llama(prompt, smoker_context, model_name=model_name)
                    if response and response.status_code == 200:
                        logger.info(f"Successful response with model: {model_name}")
                        break
                except Exception as e:
                    logger.error(f"Error with model {model_name}: {e}")
            
            with st.chat_message("assistant", avatar="👤"):
                if response is None:
                    error_msg = "Failed to connect to Llama API. Check your credentials and logs."
                    st.error(error_msg)
                    ss.messages_smoker.append({"role": "assistant", "content": error_msg})
                elif response.status_code != 200:
                    error_msg = f"API error: {response.status_code} - Check logs for details"
                    st.error(error_msg)
                    ss.messages_smoker.append({"role": "assistant", "content": error_msg})
                else:
                    logger.info("Streaming Llama 3.3 response")
                    full_response = st.write_stream(stream_llama_response(response))
                    logger.info(f"Completed Llama response stream, got full response: {bool(full_response)}")
                    if full_response:
                        # Add to message history
                        ss.messages_smoker.append({"role": "assistant", "content": full_response})
                    else:
                        st.error("Empty response received from Llama API")
                        ss.messages_smoker.append({"role": "assistant", "content": "Sorry, I received an empty response. Please try again."})

run()