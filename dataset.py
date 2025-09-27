import kagglehub
def getDataset():
        
    # Download latest version
    path = kagglehub.dataset_download("emmarex/plantdisease")

    print("Path to dataset files:", path)
    
    return path
