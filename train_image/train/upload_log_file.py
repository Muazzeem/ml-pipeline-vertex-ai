import json

import gcsfs


def store_model_info(data, project, log_path):
    print(log_path)
    fs = gcsfs.GCSFileSystem(project=project)
    start_index = log_path.find("/", 1) + 1
    end_index = log_path.find("/", start_index)
    bucket_name = log_path[start_index:end_index]

    json_object = json.dumps(data, indent=4)
    with fs.open(path=f"{bucket_name}/models/latest/logs/model_info.json", mode='w') as f:
        f.write(json_object)
