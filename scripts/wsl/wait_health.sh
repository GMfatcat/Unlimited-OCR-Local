#!/usr/bin/env bash
# 輪詢 SGLang server /health，最多約 180s。
URL="http://127.0.0.1:10000/health"
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "$URL" || true)
  if [ "$code" = "200" ]; then
    echo "READY after ${i} polls (code 200)"
    exit 0
  fi
  sleep 3
done
echo "NOT_READY last_code=${code:-none}"
exit 1
