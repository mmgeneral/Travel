import openai
import numpy as np
import copy
import ast
import re
import math
import time
import os
import requests
import anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

claude_api_key_name = ...# your key
mixtral_api_key_name = ...# your key

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_URL = 'https://api.deepseek.com/v1/chat/completions'

# 原本 GPT-4 的模型名索引，一概改路由到 DeepSeek，方便不动其他呼叫結構
DEEPSEEK_MODEL_MAP = {
    'gpt-4-turbo-preview': 'deepseek-chat',
    'gpt-4-1106-preview': 'deepseek-chat',
    'gpt-4': 'deepseek-chat',
    'gpt-4o': 'deepseek-chat',
    'gpt-4-32k': 'deepseek-chat',
    'gpt-3.5-turbo-0301': 'deepseek-chat',
    'gpt-4-0613': 'deepseek-chat',
    'gpt-4-32k-0613': 'deepseek-chat',
    'gpt-3.5-turbo-16k-0613': 'deepseek-chat',
    'gpt-3.5-turbo': 'deepseek-chat',
}

def DeepSeek_response(messages, model_name='deepseek-chat'):
  """用 DeepSeek（OpenAI 相容介面）取代原本的 OpenAI GPT-4 呼叫。
  保留與原本 GPT_response 同樣的 messages 輸入格式（一段字串，被包進 user role）。
  """
  if not DEEPSEEK_API_KEY:
    raise RuntimeError('DEEPSEEK_API_KEY 未設定，請在環境変量裡設定後再跳。')
  headers = {
      'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
      'Content-Type': 'application/json',
  }
  payload = {
      'model': model_name,
      'messages': [
          {'role': 'system', 'content': 'You are a helpful assistant.'},
          {'role': 'user', 'content': messages},
      ],
      'temperature': 0.0,
      'top_p': 1,
  }
  resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=120)
  resp.raise_for_status()
  data = resp.json()
  return data['choices'][0]['message']['content']

def GPT_response(messages, model_name):
  if model_name in DEEPSEEK_MODEL_MAP:
    # 原本走 OpenAI GPT-4，現在改路由到 DeepSeek，保持呼叫簽名不变
    return DeepSeek_response(messages, DEEPSEEK_MODEL_MAP[model_name])
  if model_name in ['gpt-4-turbo-preview','gpt-4-1106-preview', 'gpt-4', 'gpt-4o', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo']:
    #print(f'-------------------Model name: {model_name}-------------------')
    response = openai.ChatCompletion.create(
      model=model_name,
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": messages}
        ],
      temperature = 0.0,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    
  return response.choices[0].message.content

def Claude_response(messages):
  client = anthropic.Anthropic(
    api_key=claude_api_key_name,
  )
  message = client.messages.create(
    model="claude-3-opus-20240229", # claude-3-sonnet-20240229, claude-3-opus-20240229, claude-3-haiku-20240307
    max_tokens=4096,
    temperature=0.0,
    system="",
    messages=[
        {"role": "user", "content": messages}
    ]
  )
  return message.content[0].text

def Mixtral_response(messages, mode = 'normal'):
  model = 'mistral-large-latest'
  client = MistralClient(api_key=mixtral_api_key_name)

  if mode == 'json':
    messages = [
        ChatMessage(role="system", content="You are a helpful code assistant. Your task is to generate a valid JSON object based on the given information. Please only produce the JSON output and avoid explaining."), 
        ChatMessage(role="user", content=messages)
    ]
  elif mode == 'code':
    messages = [
    ChatMessage(role="system", content="You are a helpful code assistant that help with writing Python code for a user requests. Please only produce the function and avoid explaining. Do not add \ in front of _"),
    ChatMessage(role="user", content=messages)
  ]
  else: 
    messages = [
    ChatMessage(role="user", content=messages)
  ]

  # No streaming
  chat_response = client.chat(
      model=model,
      messages=messages,
      temperature=0.0,
  )
  # import pdb; pdb.set_trace()
  return chat_response.choices[0].message.content

