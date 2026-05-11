import yt_dlp
import os

downloads = [
    # 1st
    # {"url": "https://youtu.be/vASaBrY5PMg?si=1IXcWDE06BWwhkfV", "name": "kenjiro_tsuda"},
    # 2nd
    # {"url": "https://youtu.be/X2V4p-R4Qfo?si=uqvO7o_VK3gpGjfF", "name": "kenjiro_tsuda"},
    {"url": "https://youtu.be/zD3vxXyvOQk?si=RHfrcqZpCfEW2Auy", "name": "kenjiro_tsuda"},

    
    # {"url": "https://youtu.be/aH_irEzBOOI?si=56DkgTUatNgbqQPm", "name": "megumi_hayashibara"},
    # {"url": "https://youtu.be/iR3sfo-KtoI?si=VRRZGFbDsKipS_5k", "name": "megumi_hayashibara"},


    # {"url": "https://youtu.be/c2FalYYVDaQ?si=IB_wLnjiuzO-lONP", "name": "abi"}
   ]

for item in downloads:
    folder = f"voices/{item['name']}"
    os.makedirs(folder, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{folder}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([item["url"]])

print("Done!")