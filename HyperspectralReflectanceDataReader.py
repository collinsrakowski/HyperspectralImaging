# import spectral

# # Replace 'file_name.dat' with the path to your ENVI-readable file
# file_name = 'REFLECTANCE_emptyname_2023-07-20_19-11-39'
# dat_name = 'REFLECTANCE_emptyname_2023-07-20_19-11-39.dat'

# # Open the ENVI-format file
# img = spectral.envi.open(file_name + '.hdr', dat_name)

# # Load the data into a Python array
# data = img.load()

# # 'data' now contains the hyperspectral cube as a 3D NumPy array
# print(data.shape)  # Print the shape of the cube (lines, samples, bands)

import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from spectral import open_image
import spectral

# Assuming 'cube' is a 3D NumPy array representing the hyperspectral cube
file_name = 'REFLECTANCE_emptyname_2023-07-20_19-11-39'
dat_name = 'REFLECTANCE_emptyname_2023-07-20_19-11-39.dat'

# # Open the ENVI-format file
img = spectral.envi.open(file_name + '.hdr', dat_name)
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
img_plot = ax_img.imshow(cube[:, :, 0], cmap='gray')
ax_img.set_title("Band 1 - {:.2f} nm".format(wavelengths[0]))
plt.colorbar(img_plot, ax=ax_img)

# Global variable to store the event object
current_event = None

# Function to update the displayed band and the hyperspectral response graph when the slider value changes
def update_band(*args):
    global current_event

    band_index = int(band_slider.get())
    wavelength = wavelengths[band_index]
    band_label.config(text=f"Band {band_index + 1} - {wavelength:.2f} nm")
    img_plot.set_data(cube[:, :, band_index])
    ax_img.set_title(f"Band {band_index + 1} - {wavelength:.2f} nm")

    if current_event is not None:
        # Update the spectral response graph for the selected pixel
        x_coord = int(round(current_event.xdata))
        y_coord = int(round(current_event.ydata))
        ax_response.clear()
        ax_response.plot(wavelengths, cube[y_coord, x_coord, :], label='Pixel Spectral Response')
        ax_response.set_xlabel('Wavelength (nm)')
        ax_response.set_ylabel('Reflectance')
        ax_response.legend()

    fig.canvas.draw()

# Function to handle mouse clicks on the image
def onclick(event):
    global current_event
    current_event = event
    update_band()  # Update the spectral response graph for the selected band

# Attach the onclick event to the figure
fig.canvas.mpl_connect('button_press_event', onclick)

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
