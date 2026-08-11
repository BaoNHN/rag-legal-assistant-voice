# start_ngrok.py
# Exposes this app (port 8000) publicly for a demo -- a temporary stand-in until
# real deployment (see thesis future-work notes). NOT for production use: no auth
# in front of ngrok's tunnel beyond whatever this app itself enforces.
#
# The authtoken previously hardcoded here leaked into git history (this repo's
# initial commit + a later checkpoint, already pushed to GitHub) -- rotate/revoke
# it on https://dashboard.ngrok.com/tunnels/authtokens, since removing it from
# this file does not remove it from history. Set the new token via:
#   Windows:      set NGROK_AUTHTOKEN=your_new_token
#   macOS/Linux:  export NGROK_AUTHTOKEN=your_new_token
import os
import sys
import time

import ngrok

AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
if not AUTHTOKEN:
    print("NGROK_AUTHTOKEN environment variable is not set.", file=sys.stderr)
    print("Get a token from https://dashboard.ngrok.com/tunnels/authtokens and set it first:", file=sys.stderr)
    print("  Windows:      set NGROK_AUTHTOKEN=your_token", file=sys.stderr)
    print("  macOS/Linux:  export NGROK_AUTHTOKEN=your_token", file=sys.stderr)
    sys.exit(1)

listener = ngrok.forward(8000, authtoken=AUTHTOKEN)
print(f"\n✅ Public URL: {listener.url()}\n")
print("Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")
