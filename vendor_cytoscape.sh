#!/usr/bin/env bash
set -euo pipefail

mkdir -p static/vendor

if command -v npm >/dev/null 2>&1; then
  echo "[*] Using npm to fetch cytoscape..."
  if [ ! -f package.json ]; then
    npm init -y >/dev/null
  fi
  npm i cytoscape >/dev/null
  cp node_modules/cytoscape/dist/cytoscape.min.js static/vendor/cytoscape.min.js
    cp node_modules/cytoscape/dist/cytoscape.min.js static/cytoscape.min.js

  echo "[+] Vendored to static/vendor/cytoscape.min.js"
  exit 0
fi

if command -v curl >/dev/null 2>&1; then
  echo "[*] Using curl to fetch cytoscape..."
  curl -L -o static/vendor/cytoscape.min.js https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js
  curl -L -o static/cytoscape.min.js https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js

  echo "[+] Vendored to static/vendor/cytoscape.min.js"
  exit 0
fi

echo "[!] Neither npm nor curl found."
echo "    Please download Cytoscape.js manually and place it at static/vendor/cytoscape.min.js"
exit 1
