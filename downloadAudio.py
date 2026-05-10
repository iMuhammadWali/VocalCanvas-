import yt_dlp
import os

# Add your URLs and VA names here
downloads = [
    {"url": "https://youtu.be/E0ScLiT6-HM?si=7pGHchsvcNpQVB9o", "name": "nanami"},
    {"url": "https://youtu.be/TJPGuy-vO6Q?si=I3gb5dBzakpP5iiT", "name": "gojo"},
]

for item in downloads:
    folder = f"voices/{item['name']}"
    os.makedirs(folder, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{folder}/%(title)s.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([item["url"]])

print("Done!")