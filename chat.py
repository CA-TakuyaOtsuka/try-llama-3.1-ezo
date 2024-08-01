import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

model_id = "HODACHI/Llama-3.1-8B-EZO-1.1-it"

default_prompt = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、原則日本語で回答してください。"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

streamer = TextStreamer(
    tokenizer=tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)


def sendMessage(message, history = None):
    # メッセージリストを生成
    messages = []
    if (history is not None) and (len(history) > 0):
        messages.extend(history)
    else:
        messages.append({"role": "system", "content": default_prompt})
    messages.append({"role": "user", "content": message})
    
    # メッセージリストをトークン化
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

    start = time.process_time()
    # モデルに入力
    output_ids = model.generate(
        **inputs,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
    )
    end = time.process_time()

    # モデルの出力をデコード
    output_tokens = output_ids[0][inputs["input_ids"].size(1):]
    output = tokenizer.decode(output_tokens, skip_special_tokens=True)
    messages.append({"role": "assistant", "content": output})
    
    ##
    input_token_count = inputs["input_ids"].size(1)
    output_token_count = len(output_tokens)
    total_time = end - start
    tps = output_token_count / total_time
    print("-----------------------")
    print(f"prompt tokens = {input_token_count:.7g}")
    print(f"output tokens = {output_token_count:.7g} ({tps:f} [tps])")
    print(f"   total time = {total_time:f} [s]")
    return messages


history = []
print("> ", end="")
for message in sys.stdin:
    if message == "exit\n":
        break
    history = sendMessage(message, history)
    print("> ", end="")
