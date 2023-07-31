import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from spectral import open_image
import spectral
import torch
import hyde
import os

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

# Function to update the displayed folder and its images and graphs
def update_folder(folder_name):
    global cube_radiance, cube_reflectance, wavelengths_radiance, wavelengths_reflectance

    file_name_radiance = os.path.join(base_folder_directory, folder_name, "capture", f"{folder_name}.hdr")
    file_name_reflectance = os.path.join(base_folder_directory, folder_name, "capture", f"REFLECTANCE_{folder_name}.hdr")
    dat_name_reflectance = os.path.join(base_folder_directory, folder_name, "capture", f"REFLECTANCE_{folder_name}.dat")

    # Load the hyperspectral cube using Spectral Python for radiance data
    img_radiance = open_image(file_name_radiance)
    cube_radiance = img_radiance.load()
    print("Shape of Radiance Image:", cube_radiance.shape) 

    # Load the hyperspectral cube using Spectral Python for reflectance data
    img_reflectance = spectral.envi.open(file_name_reflectance, dat_name_reflectance)
    cube_reflectance = img_reflectance.load()
    print("Shape of Reflectance Image:", cube_reflectance.shape)

    # Rotate the cubes to the right by 90 degrees
    cube_radiance = np.rot90(cube_radiance, k=-1, axes=(0, 1))
    cube_reflectance = np.rot90(cube_reflectance, k=-1, axes=(0, 1))

    # Access the wavelength information from the Spectral Python object
    wavelengths_radiance = img_radiance.bands.centers
    wavelengths_reflectance = img_reflectance.bands.centers

    # Set the initial band index and update the GUI
    band_slider.set(0)
    update_band()
    
    # Update the image plots with the new cube data
    img_plot_radiance.set_array(cube_radiance[:, :, 0])
    img_plot_reflectance.set_array(cube_reflectance[:, :, 0])

    # Update the titles of the images with the new wavelength information
    ax_img_radiance.set_title("Radiance - Band 1 - {:.2f} nm".format(wavelengths_radiance[0]))
    ax_img_reflectance.set_title("Reflectance - Band 1 - {:.2f} nm".format(wavelengths_reflectance[0]))

    # Update the y-axis limits for the reflectance graph to be between 0 and 20
    ax_response_reflectance.set_ylim(0, 20)
    
# Function to handle the "Load Images" button click event
def load_images():
    selected_index = folder_listbox.curselection()
    if selected_index:
        selected_folder = folder_listbox.get(selected_index[0])
        selected_folder_label.config(text=f"Selected Folder: {selected_folder}")
        print('Loading Images: ', selected_folder)
        update_folder(selected_folder)

# Function to move to the previous folder
def previous_folder():
    global current_folder_index, current_folder

    current_folder_index -= 1
    if current_folder_index < 0:
        current_folder_index = len(folder_names) - 1

    current_folder = folder_names[current_folder_index]
    print(f"Loading folder: {current_folder}")
    update_folder(current_folder)

# Function to move to the next folder
def next_folder():
    global current_folder_index, current_folder

    current_folder_index += 1
    if current_folder_index >= len(folder_names):
        current_folder_index = 0

    current_folder = folder_names[current_folder_index]
    print(f"Loading folder: {current_folder}")
    update_folder(current_folder)

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

# Get the base folder directory
base_folder_directory = '/run/user/1001/gvfs/sftp:host=jekyll.engr.uga.edu,user=atipa/home/atipa/collins_research/Repos/HyperspectralMugTesting_1_7-21-2023/FX17'

# Populate the folder_names array based on subdirectories in the base folder directory
folder_names = [folder for folder in os.listdir(base_folder_directory) if os.path.isdir(os.path.join(base_folder_directory, folder))]

# Create a label to display the selected folder name
selected_folder_label = tk.Label(root, text="", font=("Helvetica", 12))
selected_folder_label.pack(pady=5)

# Create a Listbox to display the folder names
folder_listbox = tk.Listbox(root, selectmode=tk.SINGLE, font=("Helvetica", 12), width=30)
folder_listbox.pack(pady=10, padx=5)

# Populate the folder_listbox with the folder names
for folder_name in folder_names:
    folder_listbox.insert(tk.END, folder_name)

# Create the "Load Images" button
load_button = tk.Button(root, text="Load Images", command=load_images)
load_button.pack(pady=5)

# Embed the matplotlib figures in the tkinter window
canvas_radiance = FigureCanvasTkAgg(fig_radiance, master=root)
canvas_radiance.draw()
canvas_radiance.get_tk_widget().pack(side=tk.LEFT, padx=5)

canvas_reflectance = FigureCanvasTkAgg(fig_reflectance, master=root)
canvas_reflectance.draw()
canvas_reflectance.get_tk_widget().pack(side=tk.LEFT, padx=5)

# Initial folder index
current_folder_index = 0

# Initial folder name
current_folder = folder_names[current_folder_index]

# Update the displayed folder initially
update_folder(current_folder)

# Run the tkinter event loop
root.mainloop()

