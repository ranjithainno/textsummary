# Use an official Python runtime as a parent image
FROM python:3.10.2-slim

# Set the working directory in the container
WORKDIR /app

# Install SentencePiece library
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
 && rm -rf /var/lib/apt/lists/* \
 && git clone https://github.com/google/sentencepiece.git \
 && cd sentencepiece \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make -j $(nproc) \
 && make install \
 && ldconfig \
 && cd ../.. \
 && rm -rf sentencepiece

# Copy the model directory into the container at /app/model
COPY model /Lamini

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
