#!/bin/bash

# Set environment variables
export PYTHONPATH="$PYTHONPATH"
export API_KEY="INSERT_YOUR_SECURITY_API_KEY"
# For Deepseek
export DEEPSEEK_API_KEY="INSERT_YOUR_API_KEY"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"

# For OpenAI
export OPENAI_API_KEY="INSERT_YOUR_API_KEY"
export OPENAI_MODEL="4o-mini"

export ALLOWED_IPS="127.0.0.1"  # Frontend server public IP
export RATE_LIMIT="15/minute"
export GENERATION_TIMEOUT=45
export PRELOAD_MODELS="true"  # Preload all models at startup

# Start the server
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 2 \
  ${SSL_CERT_PATH:+--ssl-certfile $SSL_CERT_PATH} \
  ${SSL_KEY_PATH:+--ssl-keyfile $SSL_KEY_PATH} \
  --timeout-keep-alive 60