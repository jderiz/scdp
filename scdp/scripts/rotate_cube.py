import numpy as np
import torch


def rotate_cube_file(input_file: str, output_file: str):
    """
    Rotates the coordinates in a cube file by 90 degrees clockwise around the x-axis.
    This includes rotating both atom coordinates and cell vectors.

    Args:
        input_file (str): Path to the input cube file
        output_file (str): Path to save the rotated cube file
    """
    # Read the cube file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Parse header information
    n_atoms = int(lines[2].split()[0])
    origin = np.array([float(x) for x in lines[2].split()[1:4]])

    # Parse cell vectors
    cell = []
    for i in range(3, 6):
        cell.append([float(x) for x in lines[i].split()[1:4]])
    cell = np.array(cell)

    # Parse atom coordinates
    atom_coords = []
    for i in range(6, 6 + n_atoms):
        atom_coords.append([float(x) for x in lines[i].split()[2:5]])
    atom_coords = np.array(atom_coords)

    # Create rotation matrix (clockwise around x-axis)
    rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    # Rotate coordinates and cell vectors
    atom_coords_rot = atom_coords @ rotation
    cell_rot = cell @ rotation

    # Write the rotated cube file
    with open(output_file, "w") as f:
        # Write first two lines unchanged
        f.write(lines[0])
        f.write(lines[1])

        # Write number of atoms and origin
        f.write(f"{n_atoms:4}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}\n")

        # Write rotated cell vectors
        for i in range(3):
            f.write(f"{int(lines[i+3].split()[0]):4}")
            for j in range(3):
                f.write(f"{cell_rot[i][j]:12.6f}")
            f.write("\n")

        # Write rotated atom coordinates
        for i in range(n_atoms):
            atom_type = int(lines[i + 6].split()[0])
            f.write(f"{atom_type:4}{0.0:12.6f}")
            for j in range(3):
                f.write(f"{atom_coords_rot[i][j]:12.6f}")
            f.write("\n")

        # Write the density values unchanged
        for i in range(6 + n_atoms, len(lines)):
            f.write(lines[i])


if __name__ == "__main__":
    rotate_cube_file("caffeine_rot.cube", "caffeine_rev_rot.cube")
