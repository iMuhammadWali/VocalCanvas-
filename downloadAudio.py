import yt_dlp
import os

downloads = [
    {"url": "https://youtu.be/fOpM_qtgJlM?si=Q-t78DE0tRj3mO_B", "name": "yuichi_nakamura"},
    # {"url": "https://youtu.be/TJPGuy-vO6Q?si=I3gb5dBzakpP5iiT", "name": "gojo"},
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