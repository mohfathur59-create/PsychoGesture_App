import os

def cari_gerakan_avatar(teks):
    """
    Mencari file video di assets/videos/ berdasarkan teks label.
    """
    video_folder = os.path.join("assets", "videos")
    # Video standby kalau nggak ketemu
    video_default = "https://www.w3schools.com/html/mov_bbb.mp4" 
    
    if not teks:
        return video_default
    
    # Bersihkan teks (kecilkan huruf, buang spasi)
    nama_file = f"{teks.strip().lower()}.mp4"
    video_path = os.path.join(video_folder, nama_file)
    
    if os.path.exists(video_path):
        return video_path
    else:
        # Jika file fisik ga ada, balik ke default
        return video_default