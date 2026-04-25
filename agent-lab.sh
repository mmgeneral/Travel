#!/bin/bash
# /ssd/mmg/travel/agent-lab.sh

# 1. Build the workspace
echo "🏗️  Building Hardened Workspace..."
docker build -t travel-agent-lab -f Dockerfile.workspace . || exit 1

# 2. GENERATE MOCK IDENTITY (The Staff Move)
# We create a local passwd file that includes YOUR high UID
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(whoami)
MOCK_PASSWD="$(pwd)/.passwd.mock"

# Copy the host passwd and append your record if it's missing
cat /etc/passwd > "$MOCK_PASSWD"
if ! grep -q ":$USER_ID:" "$MOCK_PASSWD"; then
    echo "$USER_NAME:x:$USER_ID:$GROUP_ID:$USER_NAME:/home/agent:/bin/bash" >> "$MOCK_PASSWD"
fi

echo "🛡️  Entering Fortress as $USER_NAME ($USER_ID)..."

# 3. RUN WITH IDENTITY INJECTION
docker run -it --rm \
  --name travel_agent_workspace \
  --network travel_default \
  --security-opt="no-new-privileges:true" \
  --user $USER_ID:$GROUP_ID \
  -v "$MOCK_PASSWD":/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $(pwd):/home/agent/app \
  -v ~/.config/claude-code:/home/agent/.config/claude-code \
  -v ~/.gemini:/home/agent/.gemini \
  -e HOME=/home/agent \
  -e USER=$USER_NAME \
  -e LOGNAME=$USER_NAME \
  -w /home/agent/app \
  travel-agent-lab
