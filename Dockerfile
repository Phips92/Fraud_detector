FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the Conda environment definition
COPY environment.yml .

# Create the Conda environment 
RUN conda env create -f environment.yml

# Use the Conda environment for all subsequent commands
SHELL ["conda", "run", "-n", "fraud_detect_env", "/bin/bash", "-c"]

# Copy the rest of the application code
COPY . .

# Expose the Flask port
EXPOSE 5000

# Start the application using the Conda environment
CMD ["conda", "run", "-n", "fraud_detect_env", "python", "app.py"]

