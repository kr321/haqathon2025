# # chatbot.py
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load a quantized or small LLaMA model locally
# model_name = "TheBloke/Llama-2-7B-Chat-GGML"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)


# def get_response(detected_objects):
#     object_list = ", ".join(detected_objects)
#     prompt = (
#         f"You are a disaster relief assistant. Based on the following objects detected: {object_list}, "
#         "give specific survival or rescue advice for the user."
#     )

#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_length=200)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


from llama_cpp import Llama

# Path to your local ggml quantized model file (download manually)
model_path = "models/ggml-model-q4_0.bin"  # update path accordingly

llm = Llama(model_path=model_path)

def get_response(detected_objects):
    object_list = ", ".join(detected_objects)
    prompt = (
        f"You are a disaster relief assistant. Based on the following objects detected: {object_list}, "
        "give specific survival or rescue advice for the user."
    )
    output = llm(prompt, max_tokens=200)
    return output['choices'][0]['text'].strip()
