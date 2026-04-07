#!/bin/bash
cd "$(dirname "$0")"

# Clear stale __pycache__ from installed extension
EXT_DIR="$HOME/.config/blender/5.1/extensions/user_default/blendgmsh"
if [ -d "$EXT_DIR" ]; then
    find "$EXT_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
fi

rm -rf /tmp/blendgmsh /tmp/blendgmsh.zip
cp -r blender_addon /tmp/blendgmsh
cp -r step_converter /tmp/blendgmsh/step_converter
cp -r matching_library /tmp/blendgmsh/matching_library
cp -r schema /tmp/blendgmsh/schema
find /tmp/blendgmsh -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
cd /tmp && zip -r /tmp/blendgmsh.zip blendgmsh/
rm -rf /tmp/blendgmsh
echo "/tmp/blendgmsh.zip"
