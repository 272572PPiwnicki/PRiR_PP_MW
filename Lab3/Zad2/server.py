import socket
from PIL import Image
import numpy as np
from net_utils import send_all, receive_all

def split_image_to_stripes(img_array: np.ndarray, n_parts: int):
    h, w = img_array.shape[0], img_array.shape[1]
    stripe_height = h // n_parts
    fragments = []
    for i in range(n_parts):
        start = i * stripe_height
        end = (i + 1) * stripe_height if i < n_parts - 1 else h
        fragments.append(img_array[start:end, :, :])
    return fragments

def merge_stripes(stripes):
    return np.vstack(stripes)

def server_main(image_path, n_clients, bind_ip="0.0.0.0", bind_port=2040):
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    fragments = split_image_to_stripes(img_array, n_clients)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((bind_ip, bind_port))
    server_socket.listen(n_clients)
    print("Serwer nasłuchuje ...", bind_ip, bind_port)

    processed_fragments = []

    for i in range(n_clients):
        client_socket, client_address = server_socket.accept()
        print(f"Połączono z klientem {i+1}: {client_address}")

        # wyślij fragment i odbierz przetworzony
        send_all(client_socket, fragments[i])
        processed_fragment = receive_all(client_socket)
        processed_fragments.append(processed_fragment)

        client_socket.close()

    # scalenie
    result_image = merge_stripes(processed_fragments)
    out_img = Image.fromarray(result_image)
    out_img.save("processed_image.png")
    print("Obraz przetworzony zapisany jako processed_image.png")
    server_socket.close()

if __name__ == "__main__":
    server_main("input.jpg", n_clients=1, bind_ip="192.168.1.106", bind_port=2040)