import socket
import numpy as np
from net_utils import send_all, receive_all

# ten sam filtr co wcześniej – można skopiować
def sobel_edge_detect_gray(fragment_gray: np.ndarray) -> np.ndarray:
    Kx = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]], dtype=float)
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=float)

    h, w = fragment_gray.shape
    G = np.zeros((h, w), dtype=np.float32)
    for y in range(1, h-1):
        for x in range(1, w-1):
            region = fragment_gray[y-1:y+2, x-1:x+2]
            gx = np.sum(Kx * region)
            gy = np.sum(Ky * region)
            G[y, x] = np.sqrt(gx*gx + gy*gy)

    G = (G / G.max()) * 255.0 if G.max() != 0 else G
    return G.astype(np.uint8)

def edge_filter(fragment: np.ndarray) -> np.ndarray:
    # fragment przyjdzie najprawdopodobniej jako RGB (H, W, 3)
    if fragment.ndim == 3:
        gray = (
            0.299 * fragment[:, :, 0]
            + 0.587 * fragment[:, :, 1]
            + 0.114 * fragment[:, :, 2]
        ).astype(np.float32)
    else:
        gray = fragment.astype(np.float32)

    edges = sobel_edge_detect_gray(gray)
    # zwrócimy 3-kanał, żeby serwer mógł składać
    edges_rgb = np.stack([edges, edges, edges], axis=2)
    return edges_rgb

def client_main(server_ip="192.168.1.3", server_port=2040):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    print("Połączono z serwerem")

    fragment = receive_all(client_socket)
    processed_fragment = edge_filter(fragment)
    send_all(client_socket, processed_fragment)
    client_socket.close()

    print("Fragment przetworzony i wysłany z powrotem do serwera")

if __name__ == "__main__":
    client_main("192.168.1.106", 2040)