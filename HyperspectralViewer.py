import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from spectral import open_image
import spectral
import os
import torch
import hyde

# Assuming 'cube' is a 3D NumPy array representing the hyperspectral cube
file_name_radiance = 'capture/mug_emptyname_0004' # 'capture/emptyname_2023-07-20_19-11-39'
file_name_reflectance = 'capture/REFLECTANCE_mug_emptyname_0004' # 'REFLECTANCE_emptyname_2023-07-20_19-11-39'
dat_name_reflectance = 'capture/REFLECTANCE_mug_emptyname_0004.dat'

# Load the hyperspectral cube using Spectral Python for radiance data
img_radiance = open_image(file_name_radiance + '.hdr')
cube_radiance = img_radiance.load()

# Load the hyperspectral cube using Spectral Python for reflectance data
img_reflectance = spectral.envi.open(file_name_reflectance + '.hdr', dat_name_reflectance)
cube_reflectance = img_reflectance.load()

# Rotate the cubes to the right by 90 degrees
cube_radiance = np.rot90(cube_radiance, k=-1, axes=(0, 1))
cube_reflectance = np.rot90(cube_reflectance, k=-1, axes=(0, 1))

# Replace 'num_bands' with the total number of bands in the hyperspectral cube
num_bands_radiance = cube_radiance.shape[2]
num_bands_reflectance = cube_reflectance.shape[2]

# Access the wavelength information from the Spectral Python object
wavelengths_radiance = img_radiance.bands.centers
wavelengths_reflectance = img_reflectance.bands.centers

# Create a tkinter window
root = tk.Tk()
root.title("Hyperspectral Band Viewer")

# Create a label to display the current band index and wavelength
band_label = tk.Label(root, text="Band 1 - {:.2f} nm".format(wavelengths_radiance[0]), font=("Helvetica", 16))
band_label.pack(pady=10)

# Create matplotlib figures to display the radiance and reflectance images and the hyperspectral response graphs
fig_radiance, (ax_img_radiance, ax_response_radiance) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
img_plot_radiance = ax_img_radiance.imshow(cube_radiance[:, :, 0], cmap='inferno')
ax_img_radiance.set_title("Radiance - Band 1 - {:.2f} nm".format(wavelengths_radiance[0]))
colorbar_radiance = plt.colorbar(img_plot_radiance, ax=ax_img_radiance)
colorbar_radiance.set_label('Intensity')

fig_reflectance, (ax_img_reflectance, ax_response_reflectance) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
img_plot_reflectance = ax_img_reflectance.imshow(cube_reflectance[:, :, 0], cmap='gray')
ax_img_reflectance.set_title("Reflectance - Band 1 - {:.2f} nm".format(wavelengths_reflectance[0]))
colorbar_reflectance = plt.colorbar(img_plot_reflectance, ax=ax_img_reflectance)
colorbar_reflectance.set_label('Reflectance')

# Set the y-axis limits for the reflectance graph to be between 0 and 20
ax_response_reflectance.set_ylim(0, 30)

# Global variable to store the event object
current_event = None

def denoise_radiance_hyres(cube_radiance, band_index):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cube_radiance = cube_radiance.copy()
    print("Cube radiance shape:", cube_radiance.shape)

    # Create a new array to store the denoised cube
    denoised_cube_radiance = np.empty_like(cube_radiance)

    # Calculate the batch indices for denoising (2 bands centered around the selected band)
    start_index = max(0, band_index - 1)
    end_index = min(num_bands_radiance - 1, band_index + 1)

    # Denoise the selected bands with a batch size of 2
    for i in range(start_index, end_index + 1):
        batch_input = torch.tensor(cube_radiance[:, :, i:i + 2], dtype=torch.float32, device=device)
        hyres = hyde.FastHyDe()
        batch_output = hyres(batch_input)
        denoised_cube_radiance[:, :, i:i + 2] = batch_output.cpu().numpy()

    return denoised_cube_radiance

# Function to update the displayed band and the hyperspectral response graphs when the slider value changes
def update_band(*args):
    global current_event

    band_index = int(band_slider.get())
    wavelength_radiance = wavelengths_radiance[band_index]
    wavelength_reflectance = wavelengths_reflectance[band_index]

    band_label.config(text=f"Band {band_index + 1} - {wavelength_radiance:.2f} nm")

    # Apply denoising to the radiance cube before displaying
    # denoised_cube_radiance = denoise_radiance_hyres(cube_radiance, band_index)

    # img_plot_radiance.set_data(denoised_cube_radiance[:, :, band_index])
    img_plot_radiance.set_data(cube_radiance[:, :, band_index])
    ax_img_radiance.set_title(f"Radiance - Band {band_index + 1} - {wavelength_radiance:.2f} nm")

    img_plot_reflectance.set_data(cube_reflectance[:, :, band_index])
    ax_img_reflectance.set_title(f"Reflectance - Band {band_index + 1} - {wavelength_reflectance:.2f} nm")

    # Set the color bar range for the reflectance graph to be from 0 to 20
    img_plot_reflectance.set_clim(0, 10)

    if current_event is not None:
        # Update the spectral response graphs for the selected pixel
        x_coord = int(round(current_event.xdata))
        y_coord = int(round(current_event.ydata))
        ax_response_radiance.clear()
        ax_response_radiance.plot(wavelengths_radiance, cube_radiance[y_coord, x_coord, :], label='Pixel Spectral Response (Radiance)', color='blue')
        ax_response_radiance.set_xlabel('Wavelength (nm)')
        ax_response_radiance.set_ylabel('Intensity')
        ax_response_radiance.legend()

        ax_response_reflectance.clear()
        ax_response_reflectance.plot(wavelengths_reflectance, cube_reflectance[y_coord, x_coord, :], label='Pixel Spectral Response (Reflectance)', color='red')
        ax_response_reflectance.set_xlabel('Wavelength (nm)')
        ax_response_reflectance.set_ylabel('Reflectance')
        ax_response_reflectance.legend()
        
        # Set the y-axis limits for the reflectance graph to be between 0 and 20
        ax_response_reflectance.set_ylim(0, 10)
        # ax_response_reflectance.set_xlim(380, 700)

        # Display the corresponding value for the reflectance graph based on the pixel location and wavelength
        value_reflectance = cube_reflectance[y_coord, x_coord, band_index]
        ax_response_reflectance.text(0.95, 0.95, f'Reflectance: {value_reflectance:.3f}', ha='right', va='top', transform=ax_response_reflectance.transAxes)

    fig_radiance.canvas.draw()
    fig_reflectance.canvas.draw()

# Function to handle mouse clicks on the images
def onclick(event):
    global current_event
    current_event = event
    update_band()  # Update the spectral response graphs for the selected band

# Attach the onclick event to the figures
fig_radiance.canvas.mpl_connect('button_press_event', onclick)
fig_reflectance.canvas.mpl_connect('button_press_event', onclick)

# Create a slider to select the band
band_slider = ttk.Scale(root, from_=0, to=num_bands_radiance - 1, command=update_band, orient="horizontal")
band_slider.pack(pady=10)

# Start with the first band initially
band_slider.set(0)

# Embed the matplotlib figures in the tkinter window
canvas_radiance = FigureCanvasTkAgg(fig_radiance, master=root)
canvas_radiance.draw()
canvas_radiance.get_tk_widget().pack(side=tk.LEFT, padx=5)

canvas_reflectance = FigureCanvasTkAgg(fig_reflectance, master=root)
canvas_reflectance.draw()
canvas_reflectance.get_tk_widget().pack(side=tk.LEFT, padx=5)

# Run the tkinter event loop
root.mainloop()

# ------------------------------------------------------------------------------------------------------

# import numpy as np
# import tkinter as tk
# from tkinter import ttk
# import matplotlib
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.pyplot as plt
# from spectral import open_image
# import spectral

# import torch
# import hyde
# import concurrent.futures

# # Assuming 'cube' is a 3D NumPy array representing the hyperspectral cube
# file_name_radiance = 'emptyname_2023-07-20_19-11-39'
# file_name_reflectance = 'REFLECTANCE_emptyname_2023-07-20_19-11-39'
# dat_name_reflectance = 'REFLECTANCE_emptyname_2023-07-20_19-11-39.dat'

# # Load the hyperspectral cube using Spectral Python for radiance data
# img_radiance = open_image(file_name_radiance + '.hdr')
# cube_radiance = img_radiance.load()

# # Load the hyperspectral cube using Spectral Python for reflectance data
# img_reflectance = spectral.envi.open(file_name_reflectance + '.hdr', dat_name_reflectance)
# cube_reflectance = img_reflectance.load()

# # Rotate the cubes to the right by 90 degrees
# cube_radiance = np.rot90(cube_radiance, k=-1, axes=(0, 1))
# cube_reflectance = np.rot90(cube_reflectance, k=-1, axes=(0, 1))

# # Replace 'num_bands' with the total number of bands in the hyperspectral cube
# num_bands_radiance = cube_radiance.shape[2]
# num_bands_reflectance = cube_reflectance.shape[2]

# # Access the wavelength information from the Spectral Python object
# wavelengths_radiance = img_radiance.bands.centers
# wavelengths_reflectance = img_reflectance.bands.centers

# # Create a tkinter window
# root = tk.Tk()
# root.title("Hyperspectral Band Viewer")

# # Create a label to display the current band index and wavelength
# band_label = tk.Label(root, text="Band 1 - {:.2f} nm".format(wavelengths_radiance[0]), font=("Helvetica", 16))
# band_label.pack(pady=10)

# # Create matplotlib figures to display the radiance and reflectance images and the hyperspectral response graphs
# fig_radiance, (ax_img_radiance, ax_response_radiance) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
# img_plot_radiance = ax_img_radiance.imshow(cube_radiance[:, :, 0], cmap='inferno')
# ax_img_radiance.set_title("Radiance - Band 1 - {:.2f} nm".format(wavelengths_radiance[0]))
# plt.colorbar(img_plot_radiance, ax=ax_img_radiance)

# fig_reflectance, (ax_img_reflectance, ax_response_reflectance) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
# img_plot_reflectance = ax_img_reflectance.imshow(cube_reflectance[:, :, 0], cmap='gray')
# ax_img_reflectance.set_title("Reflectance - Band 1 - {:.2f} nm".format(wavelengths_reflectance[0]))
# plt.colorbar(img_plot_reflectance, ax=ax_img_reflectance)

# # Global variable to store the event object
# current_event = None

# def denoise_radiance_hyres(cube_radiance, band_index):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     cube_radiance = cube_radiance.copy()
#     print("Cube radiance shape:", cube_radiance.shape)

#     # Create a new array to store the denoised cube
#     denoised_cube_radiance = np.empty_like(cube_radiance)

#     # Calculate the batch indices for denoising (2 bands centered around the selected band)
#     start_index = max(0, band_index - 1)
#     end_index = min(num_bands_radiance - 1, band_index + 1)

#     # Denoise the selected bands with a batch size of 2
#     for i in range(start_index, end_index + 1):
#         batch_input = torch.tensor(cube_radiance[:, :, i:i + 2], dtype=torch.float32, device=device)
#         hyres = hyde.FastHyDe()
#         batch_output = hyres(batch_input)
#         denoised_cube_radiance[:, :, i:i + 2] = batch_output.cpu().numpy()

#     return denoised_cube_radiance

# # Function to denoise the entire hypercube asynchronously
# def denoise_entire_hypercube():
#     # Copy the radiance cube
#     cube_radiance_copy = cube_radiance.copy()

#     # Denoise the entire cube in batches using concurrent.futures
#     num_workers = 8  # You can adjust this value based on the number of available CPU cores
#     batch_size = num_bands_radiance // num_workers
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#         futures = []
#         for i in range(0, num_bands_radiance, batch_size):
#             batch_input = torch.tensor(cube_radiance_copy[:, :, i:i + batch_size], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
#             hyres = hyde.FastHyDe()
#             futures.append(executor.submit(hyres, batch_input))

#         # Wait for all the futures to complete
#         concurrent.futures.wait(futures)

#         # Collect the results
#         for i, future in enumerate(futures):
#             start_index = i * batch_size
#             end_index = min(start_index + batch_size, num_bands_radiance)
#             batch_output = future.result().cpu().numpy()
#             cube_radiance_copy[:, :, start_index:end_index] = batch_output

#     return cube_radiance_copy

# # Function to denoise the entire hypercube and update the denoised image
# def denoise_hypercube_and_update_image():
#     # Denoise the entire hypercube asynchronously
#     denoised_cube_radiance = denoise_entire_hypercube()

#     # Update the displayed image with the denoised cube
#     band_index = int(band_slider.get())
#     img_plot_radiance.set_data(denoised_cube_radiance[:, :, band_index])
#     ax_img_radiance.set_title(f"Radiance (Denoised) - Band {band_index + 1} - {wavelengths_radiance[band_index]:.2f} nm")

#     # Redraw the canvas
#     fig_radiance.canvas.draw()

# # Function to update the displayed band and the hyperspectral response graphs when the slider value changes
# def update_band(*args):
#     global current_event

#     band_index = int(band_slider.get())
#     wavelength_radiance = wavelengths_radiance[band_index]
#     wavelength_reflectance = wavelengths_reflectance[band_index]

#     band_label.config(text=f"Band {band_index + 1} - {wavelength_radiance:.2f} nm")
    
#     # Apply denoising to the radiance cube before displaying
#     # denoised_cube_radiance = denoise_radiance_hyres(cube_radiance, band_index)

#     # img_plot_radiance.set_data(denoised_cube_radiance[:, :, band_index])
#     # ax_img_radiance.set_title(f"Radiance (Denoised) - Band {band_index + 1} - {wavelength_radiance:.2f} nm")
    
#     img_plot_radiance.set_data(cube_radiance[:, :, band_index])
#     ax_img_radiance.set_title(f"Radiance - Band {band_index + 1} - {wavelength_radiance:.2f} nm")

#     img_plot_reflectance.set_data(cube_reflectance[:, :, band_index])
#     ax_img_reflectance.set_title(f"Reflectance - Band {band_index + 1} - {wavelength_reflectance:.2f} nm")

#     if current_event is not None:
#         # Update the spectral response graphs for the selected pixel
#         x_coord = int(round(current_event.xdata))
#         y_coord = int(round(current_event.ydata))
#         ax_response_radiance.clear()
#         ax_response_radiance.plot(wavelengths_radiance, cube_radiance[y_coord, x_coord, :], label='Pixel Spectral Response (Radiance)', color='blue')
#         ax_response_radiance.set_xlabel('Wavelength (nm)')
#         ax_response_radiance.set_ylabel('Intensity')
#         ax_response_radiance.legend()

#         ax_response_reflectance.clear()
#         ax_response_reflectance.plot(wavelengths_reflectance, cube_reflectance[y_coord, x_coord, :], label='Pixel Spectral Response (Reflectance)', color='red')
#         ax_response_reflectance.set_xlabel('Wavelength (nm)')
#         ax_response_reflectance.set_ylabel('Reflectance')
#         ax_response_reflectance.legend()

#     fig_radiance.canvas.draw()
#     fig_reflectance.canvas.draw()

# # Function to handle mouse clicks on the images
# def onclick(event):
#     global current_event
#     current_event = event
#     update_band()  # Update the spectral response graphs for the selected band

# # Attach the onclick event to the figures
# fig_radiance.canvas.mpl_connect('button_press_event', onclick)
# fig_reflectance.canvas.mpl_connect('button_press_event', onclick)

# # Create a slider to select the band
# band_slider = ttk.Scale(root, from_=0, to=num_bands_radiance - 1, command=update_band, orient="horizontal")
# band_slider.pack(pady=10)

# # Start with the first band initially
# band_slider.set(0)

# # Create the denoising button
# denoise_button = tk.Button(root, text="Denoise Hypercube", command=denoise_hypercube_and_update_image)
# denoise_button.pack(pady=5)

# # Embed the matplotlib figures in the tkinter window
# canvas_radiance = FigureCanvasTkAgg(fig_radiance, master=root)
# canvas_radiance.draw()
# canvas_radiance.get_tk_widget().pack(side=tk.LEFT, padx=5)

# canvas_reflectance = FigureCanvasTkAgg(fig_reflectance, master=root)
# canvas_reflectance.draw()
# canvas_reflectance.get_tk_widget().pack(side=tk.LEFT, padx=5)

# # Run the tkinter event loop
# root.mainloop()


######################################################################################
