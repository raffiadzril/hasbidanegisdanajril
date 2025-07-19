import requests
import csv
import time

API_KEY = 'AIzaSyBj1hCqak2ovj56QxZeGGck7xvxC-9cUs4'
VIDEO_ID = 'Z0wq2ryAMM4'
URL = 'https://www.googleapis.com/youtube/v3/commentThreads'

params = {
    'part': 'snippet',
    'videoId': VIDEO_ID,
    'key': API_KEY,
    'textFormat': 'plainText',
    'maxResults': 100
}

all_comments = []

while True:
    response = requests.get(URL, params=params)
    data = response.json()

    if 'items' not in data:
        print("Terjadi error:", data)
        break

    for item in data['items']:
        comment_snippet = item['snippet']['topLevelComment']['snippet']
        comment = comment_snippet['textDisplay']
        author = comment_snippet['authorDisplayName']
        published_at = comment_snippet['publishedAt']
        like_count = comment_snippet['likeCount']
        all_comments.append([author, published_at, like_count, comment])

    print(f"✅ Mengambil {len(data['items'])} komentar... Total: {len(all_comments)}")

    if 'nextPageToken' in data:
        params['pageToken'] = data['nextPageToken']
        time.sleep(0.1)  # Hindari rate limit
    else:
        break

# Simpan ke CSV
with open('komentar_youtube-adajudol.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Author', 'Published At', 'Like Count', 'Comment'])
    writer.writerows(all_comments)

print(f"\n✅ Total komentar disimpan: {len(all_comments)} ke file 'komentar_youtube-adajudol.csv'")
