# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY src/app.py app.py
COPY models/tuned/best_svm_model.pkl models/tuned/

# Expose port and run the Flask app
EXPOSE 5000
CMD ["python", "app.py"]
