import gcsfs


def copy_model(gcs_model_path, project):
    fs = gcsfs.GCSFileSystem(project=project)
    """
    Copy a model from one GCS path to another.

    Args:
        gcs_model_path: The source GCS path of the model to be copied.

    Returns:
        str: The GCS path where the model has been copied.
    """
    model_path = gcs_model_path.replace("/gcs/", "")
    start_index = gcs_model_path.find("/", 1) + 1
    end_index = gcs_model_path.find("/", start_index)
    bucket_name = gcs_model_path[start_index:end_index]

    # Copy the model from path1 to path2
    fs.copy(path1=model_path, path2=f"{bucket_name}/models/latest/")
    return gcs_model_path
