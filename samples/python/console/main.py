from strathweb_phi_engine import *
import os

# Setup inference options
inference_options_builder = InferenceOptionsBuilder()
inference_options_builder.with_temperature(temperature=0.9)
inference_options_builder.with_seed(seed=146628346)
inference_options = inference_options_builder.build()

cache_dir = os.path.join(os.getcwd(), ".cache")

class ModelEventsHandler(PhiEventHandler):
    def on_inference_token(self, token: str):
        print(token, end="")

    def on_inference_started(self):
        pass

    def on_inference_ended(self):
        pass

    def on_model_loaded(self):
        print("""
 🧠 Model loaded!
****************************************
""")

model_builder = PhiEngineBuilder()
model_builder.with_event_handler(event_handler=ModelEventsHandler())
gpu_enabled = model_builder.try_use_gpu()
model = model_builder.build_stateful(cache_dir=cache_dir, system_instruction="You are a hockey poet")

# Run inference
result = model.run_inference(prompt_text="Write a haiku about ice hockey", inference_options=inference_options)

print(f"""

****************************************
 📝 Tokens Generated: {result.token_count}
 🖥️ Tokens per second: {result.tokens_per_second}
 ⏱️ Duration: {result.duration}s
 🏎️ GPU enabled: {gpu_enabled}
""")
