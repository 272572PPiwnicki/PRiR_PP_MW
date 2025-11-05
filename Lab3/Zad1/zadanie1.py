from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count

def sobel(fragment_array: np.ndarray) -> np.ndarray:
    # konw do szarosci, standardowe wagi rgb 
    if fragment_array.ndim == 3:
        fragment_gray = (
            0.299 * fragment_array[:, :, 0]
            + 0.587 * fragment_array[:, :, 1]
            + 0.114 * fragment_array[:, :, 2]
        )
    else:
        fragment_gray = fragment_array

    # maski sobla
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

    # normalizacja, do 0-255
    G = (G / G.max()) * 255.0 if G.max() != 0 else G
    return G.astype(np.uint8)

def split(img_array: np.ndarray, n_parts: int):
    h, w = img_array.shape[0], img_array.shape[1]
    stripe_height = h // n_parts
    fragments = []
    for i in range(n_parts):
        start = i * stripe_height
        end = (i + 1) * stripe_height if i < n_parts - 1 else h
        fragments.append(img_array[start:end, :, :])
    return fragments

def process(fragment: np.ndarray) -> np.ndarray:
    edges = sobel(fragment)
    edges_rgb = np.stack([edges, edges, edges], axis=2)
    return edges_rgb

def merge(stripes):
    return np.vstack(stripes)

def main():
    # image_path = "C:\\Users\\student\\Desktop\\PRiR_PP_MW\\koper.jpg"
    # out_path = "koper_output.png"

    # image_path = "C:\\Users\\student\\Desktop\\PRiR_PP_MW\\psiur2.jpg"
    # out_path = "psiur2_output.png"

    image_path = "C:\\Users\\student\\Desktop\\PRiR_PP_MW\\pies_je.jpg"
    out_path = "pies_je_output.png"

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    n_procs = min(cpu_count(), 4)
    fragments = split(img_array, n_procs)

    with Pool(n_procs) as pool:
        processed = pool.map(process, fragments)

    merged = merge(processed)
    out_img = Image.fromarray(merged)
    out_img.save(out_path)
    print("saved to: ", out_path)

if __name__ == "__main__":
    main()
