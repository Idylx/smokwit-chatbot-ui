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
        ss.messages_expert.append({"role": "assistant", "content": message})
        
def get_expert_personality():
    """Read the expert personality file content"""
    try:
        with open("llm_personalitites/expert.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read personality file: {e}")
        return "You are an expert in smoking cessation, helping users quit smoking."
        
def call_infomaniak_llama(prompt, expert_context, model_name="llama3"):
    """Call Infomaniak's Llama 3.3 API with the provided prompt and expert context"""
    try:
        product_id = os.getenv("INFOMANIAK_PRODUCT_ID", "")
        api_key = os.getenv("INFOMANIAK_API_KEY", "")
        
        logger.info(f"Calling Infomaniak Llama API with product ID: {product_id}")
        
        if not product_id or not api_key:
            logger.error("Infomaniak API credentials not found")
            st.error("Infomaniak API credentials not found. Please set INFOMANIAK_PRODUCT_ID and INFOMANIAK_API_KEY in .env file.")
            return None
            
        url = f"https://api.infomaniak.com/1/ai/{product_id}/openai/chat/completions"
        logger.info(f"API URL: {url}")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Request headers: {headers}")
        
        # Make sure we have the right history format for the API
        conversation_history = []
        if "messages_expert" in ss and len(ss.messages_expert) > 0:
            # Convert history to appropriate format - only include relevant context
            # Limit to last 10 messages to avoid token limits
            for msg in ss.messages_expert[-10:]:
                conversation_history.append({"role": msg["role"], "content": msg["content"]})
        
        # Always add the system message first
        messages = [{"role": "system", "content": expert_context}]
        # Add conversation history
        messages.extend(conversation_history)
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Log the messages for debugging
        logger.info(f"Total messages in request: {len(messages)}")
        for i, msg in enumerate(messages):
            logger.info(f"Message {i}: role={msg['role']}, content preview: {msg['content'][:50]}...")
        
        # Try different model names to test compatibility
        model_to_use = model_name  # Use the provided model name
        
        logger.info(f"Testing with model name: {model_to_use}")
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
            "max_tokens": 1024  # Add a max_tokens parameter
        }
        
        payload_log = payload.copy() # Don't log the full content
        logger.info(f"Request payload: {json.dumps(payload_log)}")
        
        
        response = requests.post(url, headers=headers, json=payload, stream=True)
        logger.info(f"Received response with status code: {response.status_code}")
        
        # Log response headers
        logger.info(f"Response headers: {dict(response.headers)}")
        
        # If error, log the full response body for debugging
        if response.status_code != 200:
            try:
                # Try to read some of the response for error details
                content = response.raw.read(1024).decode("utf-8")
                logger.error(f"Error response content: {content}")
            except Exception as e:
                logger.error(f"Could not read error response: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calling Infomaniak API: {e}")
        st.error(f"Error connecting to Infomaniak API: {str(e)}")
        return None
           
        
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
    """
    Display the stream content, either within a new chat message or using the existing context
    """
    ss.stream = content_stream
    
    # Don't create another chat message if we're already in one
    if create_context:
        with st.chat_message("assistant", avatar="👩‍⚕️"):
            response = st.write_stream(data_streamer)
    else:
        # Just stream the content directly without creating a new chat message
        response = st.write_stream(data_streamer)
        
    if response is not None:
        if isinstance(response, list):
            # Multiple messages in the response
            for message in response:
                add_message_to_state_session(message)
        else:
            # Single message in response
            add_message_to_state_session(response)

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
        
def run():
    # Set page config first, before any other elements
    st.set_page_config(page_title="Dr. Tousse", layout="centered")
    
    # Assistant session state key
    if "expert_assistant" not in ss:
        assistant = client.beta.assistants.retrieve(assistant_id=os.environ["EXPERT_ID"])
        if assistant is None:
            raise RuntimeError(f"Assistant not found.")
        logger.info(f"Located assistant: {assistant.name}")
        ss["expert_assistant"] = assistant
    assistant = ss["expert_assistant"]
    
    # Load expert personality
    expert_context = get_expert_personality()

    # Create columns for title and model selection
    title_col, model_col = st.columns([3, 1])
    
    with title_col:
        # Show title
        st.title("Dr. Tousse")
    
    with model_col:
        # Dropdown for model selection next to title
        if "selected_model" not in st.session_state:
            ss.selected_model = "OpenAI"
            
        selected_model = st.selectbox(
            "Model",
            ["OpenAI", "Llama 3.3"],
            key="model_selector",
            on_change=lambda: setattr(ss, "selected_model", ss.model_selector)
        )
    
    # Add API test button
    with st.sidebar:
        st.title("API Testing")
        if st.button("Test Infomaniak API"):
            with st.spinner("Testing API..."):
                result = test_infomaniak_api()
                st.text_area("API Test Result", result, height=200)

    # Description text
    st.write(
        "Je suis Mme Tousse, Docteur et experte en cessation tabagique, ici pour vous accompagner dans votre démarche d'arrêt du tabac. Que vous soyez prêt à arrêter, en réflexion, ou simplement curieux d'en savoir plus, je suis là pour répondre à vos questions et vous donner des conseils adaptés à votre situation."
    )

    if "messages_expert" not in st.session_state:
        ss.messages_expert = []

    # Display chat messages from state
    for message in ss.messages_expert:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="👩‍⚕️"):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Use standard Streamlit chat input (no custom columns)
    prompt = st.chat_input("Vous pouvez poser une question à l'expert")
    
    # Handle user input and generate responses
    if prompt:
        # Add user message to history and display it
        with st.chat_message("user"):
            st.write(prompt)
            
        ss.messages_expert.append({"role": "user", "content": prompt})
        
        # Process based on the selected model
        if ss.model_selector == "OpenAI":
            # OpenAI processing
            if 'expert_thread' in ss:
                thread = ss['expert_thread']
            else:
                thread = client.beta.threads.create()
                logger.info(f"Created new thread: {thread.id}")
                ss['expert_thread'] = thread

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
                display_stream(stream)  # This will create the chat message context
                
                while not tool_requests.empty():
                    logger.info("Handling tool requests")
                    tool_outputs, thread_id, run_id = handle_requires_action(tool_requests.get())
                    with client.beta.threads.runs.submit_tool_outputs_stream(
                            thread_id=thread_id,
                            run_id=run_id,
                            tool_outputs=tool_outputs
                    ) as tool_stream:
                        # Create a separate chat message for tool responses
                        with st.chat_message("assistant", avatar="👩‍⚕️"):
                            display_stream(tool_stream, create_context=False)
        else:
            # Llama 3.3 processing
            logger.info("Using Llama 3.3 model")
            
            # Test different model names if needed
            model_names = ["llama-3.3-70b", "llama3"]  # Try the documented model name first
            
            response = None
            for model_name in model_names:
                try:
                    logger.info(f"Trying with model name: {model_name}")
                    response = call_infomaniak_llama(prompt, expert_context, model_name=model_name)
                    if response and response.status_code == 200:
                        logger.info(f"Successful response with model: {model_name}")
                        break
                except Exception as e:
                    logger.error(f"Error with model {model_name}: {e}")
            
            with st.chat_message("assistant", avatar="👩‍⚕️"):
                if response is None:
                    error_msg = "Failed to connect to Llama API. Check your credentials and logs."
                    st.error(error_msg)
                    ss.messages_expert.append({"role": "assistant", "content": error_msg})
                elif response.status_code != 200:
                    error_msg = f"API error: {response.status_code} - Check logs for details"
                    st.error(error_msg)
                    ss.messages_expert.append({"role": "assistant", "content": error_msg})
                else:
                    logger.info("Streaming Llama 3.3 response")
                    full_response = st.write_stream(stream_llama_response(response))
                    logger.info(f"Completed Llama response stream, got full response: {bool(full_response)}")
                    if full_response:
                        # Add to message history
                        ss.messages_expert.append({"role": "assistant", "content": full_response})
                    else:
                        st.error("Empty response received from Llama API")
                        ss.messages_expert.append({"role": "assistant", "content": "Sorry, I received an empty response. Please try again."})


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
            "stream": False  # No streaming for this test
        }
        
        logger.info(f"Testing API with non-streaming request")
        response = requests.post(url, headers=headers, json=payload)
        
        logger.info(f"Test response status: {response.status_code}")
        logger.info(f"Test response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                json_data = response.json()
                logger.info(f"Test response JSON: {json_data}")
                if 'choices' in json_data and json_data['choices']:
                    content = json_data['choices'][0]['message']['content']
                    return f"API test successful! Response: {content}"
            except Exception as e:
                logger.error(f"Error parsing test response: {e}")
                return f"API returned 200 but could not parse response: {e}"
        else:
            logger.error(f"Test API error: {response.status_code} - {response.text}")
            return f"API test failed with status {response.status_code}"
            
    except Exception as e:
        logger.error(f"Test API exception: {e}")
        return f"API test error: {e}"


run()