import os

# Tests never call a real LLM.
os.environ["LLM_PROVIDER"] = "fake"
