from pytube import YouTube
import os
from tqdm import tqdm
def download_videos(file_path, download_path):
    # Ensure the download path exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Read the list of URLs from the file
    with open(file_path, 'r') as file:
        urls = file.readlines()

    # Download each video
    for url in tqdm(urls):
        url = url.strip()
        if url:
            try:
                yt = YouTube(url)
                stream = yt.streams.get_highest_resolution()
                stream.download(download_path)
                print(f"Downloaded: {yt.title}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    file_path = 'misc/scripts/temp_video_file.txt'
    download_path = '/'
    download_videos(file_path, download_path)