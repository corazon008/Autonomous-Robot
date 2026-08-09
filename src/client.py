import socket
import time

HOST = "192.168.40.62"  # Server IP
PORT = 65000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((HOST, PORT))
    print(f"[+] Connected to {HOST}:{PORT}")

    message = "Hello from the client!"
    client.sendall(message.encode("utf-8"))
    print(f"[>] Sent: {message}")

    while True:
        time.sleep(1)  # Wait for a second before receiving data

    data = client.recv(1024)
    print(f"[<] Received: {data.decode('utf-8')}")

print("[*] Connection closed.")
