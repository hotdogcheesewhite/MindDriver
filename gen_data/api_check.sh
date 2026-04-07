#!/bin/bash

TEXT_PROMPT="$1"

# jq 会自动处理所有需要转义的字符
JSON_DATA=$(jq -n \
  --arg content "$TEXT_PROMPT" \
  '{
    model: "qwen3-235b-a22b-instruct-2507",
    messages: [
      {
        role: "user",
        content: $content
      }
    ],
    "max_tokens": 8192, 
    stream: false
  }')

curl -X POST https://idealab.alibaba-inc.com/api/openai/v1/chat/completions \
-H "Authorization: Bearer 3711cb0f837a425f8f1d4c658a8f86d6" \
-H "Content-Type: application/json" \
-d "$JSON_DATA"