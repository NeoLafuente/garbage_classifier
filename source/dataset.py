# pythorch lightning -> dataloader

import gdown
import os

def download_data():
	folder_url = "https://drive.google.com/drive/folders/1N08rtJmbM5aaA5LfYUk3nMd9Z0n4Ftqq?usp=drive_link"
	
	local_folder = "../data/raw"
	os.makedirs(local_folder, exist_ok=True)
	
	gdown.download_folder(folder_url, output=local_folder, quiet=False, use_cookies=False)

	print("Download complete.")
