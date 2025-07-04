import numpy as np
import os

def annealed_mean(prob, pts_in_hull, T=0.38):
    prob = prob ** (1 / T)
    prob /= np.sum(prob, axis=2, keepdims=True)
    ab = np.dot(prob, pts_in_hull)  # (H*W, 2)
    return ab.reshape(prob.shape[0], prob.shape[1], 2)  # (H, W, 2)

def load_pts_in_hull():
    # Get the directory of util.py
    util_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the path to the .npy file
    file_path = os.path.join(util_dir, 'pts_in_hull.npy')
    # Load the 313 quantized color bins
    pts_in_hull = np.load(file_path)  # shape (313, 2), each is an (a, b) pair
    return pts_in_hull

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    print(current_dir)
    print("Files in util folder:", os.listdir(os.path.dirname(__file__)))

    # Example usage
    prob = np.random.rand(224, 224, 313)  # Dummy probability map
    pts_in_hull = load_pts_in_hull()
    print(pts_in_hull)
    ab = annealed_mean(prob, pts_in_hull)
    print(ab.shape)  # Should be (224, 224, 2)