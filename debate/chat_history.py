# Import the BasicChatHistory class, the BasicChatMessageStore class and BasicChatHistoryStrategy class
from llama_cpp import Llama
from llama_cpp_agent.chat_history import BasicChatHistory, BasicChatMessageStore, BasicChatHistoryStrategy
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent.providers import LlamaCppPythonProvider

# Import the LlamaCppAgent class of the framework
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType

# Create an instance of the Llama class and load the model
llama_model = Llama("gguf-models\mistral-7b-instruct-v0.2.Q6_K.gguf", n_batch=1024, n_threads=10, n_gpu_layers=40)

# Create the provider by passing the Llama class instance to the LlamaCppPythonProvider class
provider = LlamaCppPythonProvider(llama_model)

# Create a message store for the chat history
chat_history_store = BasicChatMessageStore()

# Create the actual chat history, by passing the wished chat history strategy, it can be last_k_message or last_k_tokens. The default strategy will be to use the 20 last messages for the chat history.
# We will use the last_k_tokens strategy which will include the last k tokens into the chat history. When we use this strategy, we will have to pass the provider to the class.
chat_history = BasicChatHistory(message_store=chat_history_store, chat_history_strategy=BasicChatHistoryStrategy.last_k_tokens, k=7000, llm_provider=provider)

# Pass the configured chat history to
agent = LlamaCppAgent(provider,
                      system_prompt="You are a helpful assistant.",
                      chat_history=chat_history,
                      predefined_messages_formatter_type=MessagesFormatterType.CHATML)
