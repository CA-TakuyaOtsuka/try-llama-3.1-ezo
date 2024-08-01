import transformers
import torch
import time

model_id = "HODACHI/Llama-3.1-8B-EZO-1.1-it"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、原則日本語で回答してください。"},
    {"role": "user", "content": "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。"},
]

start = time.time()
outputs = pipeline(
    messages,
    max_new_tokens=512,
)
end = time.time()
print("1回目: ", end - start)
print(outputs[0]["generated_text"][-1])

start = time.time()
outputs = pipeline(
    messages,
    max_new_tokens=512,
)
end = time.time()
print("2回目: ", end - start)
print(outputs[0]["generated_text"][-1])
