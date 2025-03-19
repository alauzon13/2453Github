import pandas as pd
import os
import cv2

# Read the list of particles and their specifications from data_cleaned.csv
data_cleaned = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/data_cleaned.csv")

# Specify the source directory (tifs folder) and the output directory
source_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/tifs"
output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create a list to store the file names and classes
extracted_particles = []

# Create empty particleID col
data_cleaned['ParticleID'] = ""


# Iterate through the list of particles and extract vignettes
for index, row in data_cleaned.iterrows():
    # Get the file name and path
    file_name = row["Image.File"]
    file_path = os.path.join(source_dir, file_name)
    
    # Generate the vignette file name and path
    vignette_name = f"{os.path.splitext(file_name)[0]}_vign{index:06d}.png"
    vignette_path = os.path.join(output_dir, vignette_name)
    
    # Check if the vignette already exists
    if os.path.exists(vignette_path):
        print(f"Vignette already exists: {vignette_name}")
        continue
    
    # Check if the file exists in the source directory
    if os.path.exists(file_path):
        # Load the image
        image = cv2.imread(file_path)
        
        # Extract the vignette based on the specified coordinates and dimensions
        x = row["Image.X"]
        y = row["Image.Y"]
        h = row["Image.Height"]
        w = row["Image.Width"]
        vignette = image[y:y+h, x:x+w]

        # Create a unique particle ID
        particle_id = f"{os.path.splitext(os.path.basename(file_path))[0]}_particle_{index}"
        
        
        # Save the vignette
        cv2.imwrite(vignette_path, vignette)
        
        # Store the particle ID, file name, and class in the list
        extracted_particles.append({"ParticleID": particle_id, "Vignette": vignette_name, "Class": row["Class"]})
        
        # Add ParticleID to the original data row
        data_cleaned.at[index, 'ParticleID'] = particle_id
        
        
        print(f"Extracted vignette: {vignette_name}")
    else:
        print(f"File not found: {file_path}")

# Create a DataFrame from the list of extracted particles
extracted_particles_df = pd.DataFrame(extracted_particles)

# Save the DataFrame to a CSV file
extracted_particles_df.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/extracted_particles.csv", index=False)

data_cleaned.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/data_cleaned.csv")

print("Finished extracting vignettes and storing classes.")