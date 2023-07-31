import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from spectral import open_image
import tifffile as tiff
import torch
import spectral
import hyde

def denoise_hypercube(cube):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cube = cube.clone()  # Make a copy to avoid negative strides
    print("Cube shape:", cube.shape)

    # Convert the entire cube to a tensor
    cube_input = torch.tensor(cube, dtype=torch.float32, device=device)
    hyres = hyde.FastHyDe()
    cube_output = hyres(cube_input)
    #convert 

    return cube_output


def view_denoised_hypercube(file_name_denoised):
    # Load the denoised hyperspectral cube using Spectral Python
    img_denoised = open_image(file_name_denoised + '.hdr')
    denoised_cube = img_denoised.load()

    # Create a Tkinter application window
    root = tk.Tk()
    root.title("Denoised Hyperspectral Image Viewer")

    # Set up the Matplotlib figure for displaying images
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Slider to navigate through bands
    band_slider = tk.Scale(root, from_=0, to=denoised_cube.shape[0] - 1, orient=tk.HORIZONTAL)
    band_slider.pack()

    def update_image(band):
        ax.clear()
        ax.imshow(denoised_cube[band], cmap='gray')
        ax.set_title(f"Denoised Band {band}")
        canvas.draw()

    # Initial update of the image with the first band
    update_image(0)

    def on_slider_move(event):
        band = band_slider.get()
        update_image(band)

    band_slider.bind("<B1-Motion>", on_slider_move)

    root.mainloop()


def save_denoised_hypercube(denoised_cube, file_name):
    # Get the original image object to retrieve metadata
    img = open_image(file_name + '.hdr')

    # Save the denoised hyperspectral data as a new ENVI file
    tiff.imsave(file_name + '_Denoised.hdr', denoised_cube, metadata=img.metadata)

def main():
    # Replace 'file_name' with the path to your hyperspectral data file (without the extension)
    file_name_radiance = 'emptyname_2023-07-20_19-11-39'
    file_name_reflectance = 'REFLECTANCE_emptyname_2023-07-20_19-11-39'
    dat_name_reflectance = 'REFLECTANCE_emptyname_2023-07-20_19-11-39.dat'
    print("Starting . . .")

    print("Loading Hyperspectral cube for Radiance Data")
    # Load the hyperspectral cube using Spectral Python for radiance data
    img_radiance = open_image(file_name_radiance + '.hdr')
    cube_radiance = img_radiance.load()

    print("Loading Hyperspectral cube for Reflectance Data")
    # Load the hyperspectral cube using Spectral Python for reflectance data
    img_reflectance = spectral.envi.open(file_name_reflectance + '.hdr', dat_name_reflectance)
    cube_reflectance = img_reflectance.load()

    # Rotate the cubes to the right by 90 degrees
    cube_radiance = np.rot90(cube_radiance, k=-1, axes=(0, 1))
    cube_reflectance = np.rot90(cube_reflectance, k=-1, axes=(0, 1))

    # Set the device to "cuda:0" explicitly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading Cube to GPU")
    # Move the entire cube to the GPU device
    cube_radiance = torch.tensor(np.copy(cube_radiance), dtype=torch.float32, device=device)
    # cube_reflectance = torch.tensor(np.copy(cube_reflectance), dtype=torch.float32, device=device)

    # Denoise the radiance and reflectance hypercubes
    print('Starting Denoising of Radiance')
    denoised_cube_radiance = denoise_hypercube(cube_radiance)
    print('Starting Denoising of Reflectance')
    # denoised_cube_reflectance = denoise_hypercube(cube_reflectance)

    # # Move the denoised cubes back to CPU for saving
    # print("Moving Cubes back to Memory")
    # denoised_cube_radiance = denoised_cube_radiance.cpu().numpy()
    # # denoised_cube_reflectance = denoised_cube_reflectance.cpu().numpy()

    # # Save the denoised data to new files
    print("Saving Cubes")
    save_denoised_hypercube(denoised_cube_radiance.cpu().numpy(), file_name_radiance)
    # save_denoised_hypercube(denoised_cube_reflectance, file_name_reflectance)
    
    # Display the denoised hyperspectral image
    print("Viewing Denoised Radiance")
    file_name_denoise_radiance = 'emptyname_2023-07-20_19-11-39_Denoised'
    view_denoised_hypercube(file_name_denoise_radiance)

if __name__ == "__main__":
    main()


