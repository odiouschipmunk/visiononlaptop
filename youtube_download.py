from pytubefix import YouTube
import os
import subprocess

def download_video(url, output_path='.'):
    try:
        yt = YouTube(url)
        
        # Get the highest resolution video stream
        video_stream = yt.streams.filter(adaptive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        # Get the highest quality audio stream
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
        
        # Download video and audio streams
        video_file = video_stream.download(output_path=output_path, filename='video.mp4')
        audio_file = audio_stream.download(output_path=output_path, filename='audio.mp4')
        
        # Merge video and audio using ffmpeg
        output_file = os.path.join(output_path, yt.title + '.mp4')
        ffmpeg_path = r'C:\\Users\\dhruv\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1-full_build-shared\\bin\\ffmpeg.exe'  # Replace with the actual path to ffmpeg.exe
        command = f'"{ffmpeg_path}" -i "{video_file}" -i "{audio_file}" -c:v copy -c:a aac "{output_file}"'
        subprocess.run(command, shell=True)
        
        # Remove temporary files
        os.remove(video_file)
        os.remove(audio_file)
        
        print(f"Downloaded: {url} at resolution: {video_stream.resolution}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def download_videos_from_file(file_path, output_path='.'):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as file:
        links = file.readlines()
    
    for link in links:
        link = link.strip()
        if link:
            download_video(link, output_path)

if __name__ == "__main__":
    input_file = 'video_links.txt'  # Replace with your text file containing YouTube links
    output_directory = 'downloads4'  # Replace with your desired output directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    download_videos_from_file(input_file, output_directory)