import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import pydicom
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class KMeansSegmentationSoftware:
    def __init__(this, gui):

        this.gui = gui
        gui.title("Segmentation of DICOM images using K-Means")

        # create input widgets
        this.num_clusters_label = tk.Label(gui, text="Number of Clusters:")
        this.num_clusters_label.grid(row=0, column=0, padx=5, pady=5)
        this.num_clusters_entry = tk.Entry(gui)
        this.num_clusters_entry.grid(row=0, column=1, padx=5, pady=5)

        this.dicom_path_label = tk.Label(gui, text="DICOM Image Path:")
        this.dicom_path_label.grid(row=1, column=0, padx=5, pady=5)
        this.dicom_path_entry = tk.Entry(gui)
        this.dicom_path_entry.grid(row=1, column=1, padx=5, pady=5)

        this.browse_button = tk.Button(gui, text="Browse", command=this.browse_dicom)
        this.browse_button.grid(row=1, column=2, padx=5, pady=5)

        this.segment_button = tk.Button(gui, text="Segment", command=this.segment_image)
        this.segment_button.grid(row=2, column=1, padx=5, pady=5)


    def browse_dicom(this):
        # open file dialog to select DICOM image
        file_path = filedialog.askopenfilename()
        this.dicom_path_entry.delete(0, tk.END)
        this.dicom_path_entry.insert(0, file_path)

        #read the DICOM file
        dicom_path = this.dicom_path_entry.get()
        ds = pydicom.dcmread(dicom_path)

        #display the original dicom image without segmentation
        pixel_data = ds.pixel_array
        plt.imshow(pixel_data, cmap=plt.cm.gray)
        plt.axis('off')  
        plt.show()
        

    def segment_image(this):

        # get user inputs
        try:
            num_clusters = int(this.num_clusters_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid input for number of clusters.")
            return

        dicom_path = this.dicom_path_entry.get()

        if not dicom_path:
            messagebox.showerror("Error", "Please select a DICOM image.")
            return


        # read DICOM image

        # Read the DICOM image using the pydicom.dcmread function
        ds = pydicom.dcmread(dicom_path)
        

        #pre-processing of the dicom image

        # store the pixel data as a numpy array of floating-point values. 
        floating_values = np.float32
        pixel_data = ds.pixel_array.astype(floating_values)

        # find the maximum in the pixel data array
        max_value = np.max(pixel_data)

         # Normalize the pixel values by dividing them by the maximum pixel value. 
        normalized_pixel_data = pixel_data / max_value

        # Reshape the pixel data array into a 2D array with one column to match the expected input format for K-means clustering. 
        reshaped_data = np.reshape(normalized_pixel_data, (-1, 1))


        # perform k-means clustering

        # Create an instance of KMeans from scikit-learn with the specified number of clusters and a random state of 0
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)

        # Fit the K-means model to the pixel data using the fit method.  
        fitting_data_on_model = kmeans.fit(reshaped_data)

        # Retrieve the cluster labels from the fitted K-means model.
        # Reshape the cluster labels array back to the original shape of the DICOM image.
        
        labels = np.reshape(fitting_data_on_model.labels_, normalized_pixel_data.shape)


        # display segmented image

        # Scale the cluster labels to the range of 0 to 255 to convert them to grayscale pixel values. 

        greyscale_format = labels * 255.0
        number_of_clusters = num_clusters - 1

        #to convert the scaled cluster labels to unsigned 8-bit integers. 
        bit_8 = np.uint8(greyscale_format / number_of_clusters)

        # Create an Image object from the scaled cluster labels array using Image.fromarray. 
        segmented_image = Image.fromarray(bit_8)

        # Display the segmented image using the show method of the Image object.
        segmented_image.show()


instance_of_TK = tk.Tk()
root = instance_of_TK
gui = KMeansSegmentationSoftware(root)
root.mainloop()

