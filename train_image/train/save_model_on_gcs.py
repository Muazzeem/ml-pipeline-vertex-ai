import os


def store_model(model_dir, epoch_num):
    """
    Stores the model in Google Cloud Storage (GCS).

    Args:
        model_dir (str): The GCS or local directory where the model will be stored.
        epoch_num (int): The epoch number to include in the model file name.

    Returns:
        str: The GCS path to the stored model file.
    """
    gs_prefix = 'gs://'
    gcsfuse_prefix = '/gcs/'
    if model_dir.startswith(gs_prefix):
        model_dir = model_dir.replace(gs_prefix, gcsfuse_prefix)
        dirpath = os.path.split(model_dir)[0]
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
    gcs_model_path = os.path.join(os.path.join(model_dir, 'epoch_' + str(epoch_num) + '.pth'))
    return gcs_model_path
