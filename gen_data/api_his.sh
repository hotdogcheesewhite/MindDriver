#!/bin/bash

if [ $# -ne 11 ]; then
  echo "Usage: $0 <text_prompt> <img1> <img2> <img3> <img4> <img5> <img6> <img7> <img8> <img9> <img10>"
  exit 1
fi

TEXT_PROMPT="$1"
IMG1="$2"
IMG2="$3"
IMG3="$4"
IMG4="$5"
IMG5="$6"
IMG6="$7"
IMG7="$8"
IMG8="$9"
IMG9="${10}"
IMG10="${11}"


BASE64_IMG1=$(base64 -w 0 "$IMG1")
BASE64_IMG2=$(base64 -w 0 "$IMG2")
BASE64_IMG3=$(base64 -w 0 "$IMG3")
BASE64_IMG4=$(base64 -w 0 "$IMG4")
BASE64_IMG5=$(base64 -w 0 "$IMG5")
BASE64_IMG6=$(base64 -w 0 "$IMG6")
BASE64_IMG7=$(base64 -w 0 "$IMG7")
BASE64_IMG8=$(base64 -w 0 "$IMG8")
BASE64_IMG9=$(base64 -w 0 "$IMG9")
BASE64_IMG10=$(base64 -w 0 "$IMG10")

# 拼接成 JSON 并提交
curl -v  \
  -H "Content-Type: " \
  -H "Authorization: " \
  -d @- <<EOF
{
  "model": "qwen3-vl-235b-a22b-instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "${TEXT_PROMPT}"},
        {"type": "text", "text": "This is the current frame's six-view image of the vehicle.: CAM_FRONT:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG1}"}},
        {"type": "text", "text": "This is the image from 0.5s ago for CAM_FRONT, black images indicate no data:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG7}"}},
        {"type": "text", "text": "This is the image from 1.0s ago for CAM_FRONT, black images indicate no data:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG8}"}},
        {"type": "text", "text": "This is the image from 1.5s ago for CAM_FRONT, black images indicate no data:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG9}"}},
        {"type": "text", "text": "This is the image from 2.0s ago for CAM_FRONT, black images indicate no data:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG10}"}},
        {"type": "text", "text": "CAM_FRONT_LEFT:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG2}"}},
        {"type": "text", "text": "CAM_FRONT_RIGHT:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG3}"}},
        {"type": "text", "text": "CAM_BACK:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG4}"}},
        {"type": "text", "text": "CAM_BACK_LEFT:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG5}"}},
        {"type": "text", "text": "CAM_BACK_RIGHT:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG6}"}}
      ]
    }
  ],
  "max_tokens": 8192
}
EOF