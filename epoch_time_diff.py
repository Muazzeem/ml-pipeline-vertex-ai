from google.cloud import storage

client = storage.Client()

bucket = storage.Bucket(client, 'subtlemedical-demo-bucket')

str_folder_name_on_gcs = 'models/latest/'
blobs = bucket.list_blobs(prefix=str_folder_name_on_gcs)

# Create a list to store the creation times
creation_times = []

for blob in blobs:
    if blob.name.endswith(".pth"):
        creation_times.append((blob.name.replace(str_folder_name_on_gcs, ''), blob.time_created))

# Sort the list by creation time
sorted_times = sorted(creation_times, key=lambda x: x[1])

# Calculate the time difference between objects
for i in range(1, len(sorted_times)):
    file1, time1 = sorted_times[i - 1]
    file2, time2 = sorted_times[i]
    time_difference = time2 - time1
    print(f"{file1} and {file2}: ========> {str(time_difference)}")
