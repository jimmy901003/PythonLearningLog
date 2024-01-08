# 父類別: 播放器
class Player:
    def __init__(self):
        self.__playlist = []  # 初始化私有播放清單(私有成員)
        self._current_index = 0
        
    def add_media(self, media):
        self.__playlist.append(media)
        print(f"已從播放清單中加入：{media}")
     
    def remove_media(self, media):
        if media in self.__playlist:
            self.__playlist.remove(media)
            print(f"已從播放清單中刪除：{media}")
        else:
            print(f"找不到名稱為 {media} 的歌曲")
        
    def play(self):
        if self.__playlist: 
            media = self.__playlist[self._current_index]
            # 每次調用 play 方法皆會切換至下一首媒體
            self._current_index = (self._current_index + 1) % len(self.__playlist)
            return media
        else:
            print("播放列表是空的")

    def __str__(self):
        return f"播放列表: {', '.join(self.__playlist)}"

# 子類別: 音樂播放器  
class MusicPlayer(Player): # 繼承自Player類別
    def __init__(self):
        super().__init__() # 使用super()調用父類別的初始化函數
          
    def add_song(self, song):
        super().add_media(song)
        
    def remove_song(self, song):
        super().remove_media(song)
    
    def play(self):
        current_song = super().play()
        return f"播放音樂: {current_song}"

# 子類別: 影片播放器 
class VideoPlayer(Player): # 繼承自Player類別
    def __init__(self):
        super().__init__()
        
    def add_video(self, video):
        super().add_media(video)

    def remove_video(self, video):
        super().remove_media(video)
        
    def play(self):
        current_video = super().play()
        return f"播放影片: {current_video}"
    
if __name__ == "__main__":
    
    print('創建音樂播放器')
    music_player = MusicPlayer()
    music_player.add_song("Song 1")
    music_player.add_song("Song 2")
    print(music_player)
    print(music_player.play())
    print(music_player.play())
    music_player.remove_song("Song 1")
    print(music_player)
    
    print('創建影片播放器')
    video_player = VideoPlayer()
    video_player.add_video("Video 1")
    video_player.add_video("Video 2")
    print(video_player)
    print(video_player.play())
    print(video_player.play())
    video_player.remove_video("Video 1")
    print(video_player)