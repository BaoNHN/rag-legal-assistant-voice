import ngrok, time

listener = ngrok.forward(5000, authtoken="3DysuGmBLxinXjODh0FXP8Lb7qP_3cKmsnBHbcKmhDUumxEHy")
print(f"\n✅ Public URL: {listener.url()}\n")
print("Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")