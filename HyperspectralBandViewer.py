import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from spectral import open_image

# Assuming 'cube' is a 3D NumPy array representing the hyperspectral cube
file_name = 'emptyname_2023-07-20_19-11-39'

# Load the hyperspectral cube using Spectral Python
img = open_image(file_name + '.hdr')
cube = img.load()

# Rotate the cube to the right by 90 degrees
cube = np.rot90(cube, k=-1, axes=(0, 1))

# Replace 'num_bands' with the total number of bands in the hyperspectral cube
num_bands = cube.shape[2]

# Access the wavelength information from the Spectral Python object
wavelengths = img.bands.centers

# Create a tkinter window
root = tk.Tk()
root.title("Hyperspectral Band Viewer")

# Create a label to display the current band index and wavelength
band_label = tk.Label(root, text="Band 1 - {:.2f} nm".format(wavelengths[0]), font=("Helvetica", 16))
band_label.pack(pady=10)

# Create a matplotlib figure to display the bands and the hyperspectral response graph
fig, (ax_img, ax_response) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
img_plot = ax_img.imshow(cube[:, :, 0], cmap='inferno')
ax_img.set_title("Band 1 - {:.2f} nm".format(wavelengths[0]))
plt.colorbar(img_plot, ax=ax_img)

# Function to update the displayed band and the hyperspectral response graph when the slider value changes
def update_band(*args):
    band_index = int(band_slider.get())
    wavelength = wavelengths[band_index]
    band_label.config(text=f"Band {band_index + 1} - {wavelength:.2f} nm")
    img_plot.set_data(cube[:, :, band_index])
    ax_img.set_title(f"Band {band_index + 1} - {wavelength:.2f} nm")

    # Hyperspectral response graph
    selected_pixel = cube[:, :, band_index].mean()  # Replace this with the pixel selection method you want
    ax_response.clear()
    ax_response.plot(wavelengths, cube.mean(axis=(0, 1)), label='Mean Hyperspectral Response')
    ax_response.scatter(wavelength, selected_pixel, color='red', label='Selected Pixel')
    ax_response.set_xlabel('Wavelength (nm)')
    ax_response.set_ylabel('Intensity')
    ax_response.legend()

    fig.canvas.draw()

# Create a slider to select the band
band_slider = ttk.Scale(root, from_=0, to=num_bands - 1, command=update_band, orient="horizontal")
band_slider.pack(pady=10)

# Start with the first band initially
band_slider.set(0)

# Embed the matplotlib figure in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Run the tkinter event loop
root.mainloop()
