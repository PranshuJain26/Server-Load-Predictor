import subprocess
import time
import os
import signal
import sys

API_CODES = ["A2","A10","A66","A69","A42",
             "A12","A21","A47","A31","A45",
             "A15","A59","A18","A40","A7"]

BASE_PORT = 5000
API_PROCESSES = []

port = BASE_PORT
for api in API_CODES:
    API_PROCESSES.append({
        "name": api,
        "file": f"{api}\\app_M{api[1:]}.py",   # ONE app per API
        "port": port
    })
    port += 1

running_processes = []

def get_python_command():
    try:
        subprocess.run(['python3', '--version'], check=True, capture_output=True)
        return 'python3'
    except:
        subprocess.run(['python', '--version'], check=True, capture_output=True)
        return 'python'

def shutdown_apis(sig, frame):
    print("\nShutting down all API services...")
    for p in running_processes:
        try:
            p.terminate()
        except:
            pass
    print("All services stopped.")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_apis)
signal.signal(signal.SIGTERM, shutdown_apis)

def start_apis():
    python_cmd = get_python_command()
    print(f"Starting {len(API_PROCESSES)} API services...\n")

    for api in API_PROCESSES:
        if not os.path.exists(api['file']):
            print(f"❌ Missing: {api['file']}")
            continue

        process = subprocess.Popen([python_cmd, api['file']])
        running_processes.append(process)
        print(f"✅ Started {api['name']} on port {api['port']}")

    print("\nAll services running. Press Ctrl+C to stop.")
    while True:
        time.sleep(5)

if __name__ == "__main__":
    start_apis()
