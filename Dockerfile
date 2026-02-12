# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app
ENV PIP_ROOT_USER_ACTION=ignore   
# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install pip-tools and sync dependencies
RUN pip install --no-cache-dir pip-tools && \
    pip-sync requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Ensure the output directory exists
RUN mkdir -p output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run main.py when the container launches
CMD ["python", "main.py"]
