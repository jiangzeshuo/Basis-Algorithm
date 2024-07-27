import ssl
import socket

try:
    context = ssl.create_default_context()
    conn = context.wrap_socket(socket.socket(socket.AF_INET), server_hostname='example.com')
    conn.connect(('example.com', 443))
    print("SSL connection established")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    conn.close()
