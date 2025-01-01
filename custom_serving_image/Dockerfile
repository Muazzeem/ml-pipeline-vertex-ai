FROM python:3.9-slim


# Install dependencies
RUN python3 -m pip install --upgrade pip

COPY . .
RUN pip3 install -r requirements.txt
EXPOSE 8080
CMD ["gunicorn", "app:app", "--timeout=0", "--workers=1", "--threads=8", "--bind=0.0.0.0:8080"]
