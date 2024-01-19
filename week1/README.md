# WEEK1: 簡單的播放器類別繼承

## 介紹
在這一週的學習中，創建了一個簡單的播放器系統，利用Python的物件導向繼承概念，建立了父類別 `Player` 和兩個子類別 `MusicPlayer` 和 `VideoPlayer`。

## 程式碼解說

### Player 類別
- 父類別 `Player` 是一個基本的播放器，擁有私有成員 `__playlist` 用於儲存播放清單，以及 `_current_index` 用於追蹤當前播放位置。
- 方法：
  - `add_media`: 新增媒體到播放清單。
  - `remove_media`: 從播放清單中刪除媒體。
  - `play`: 播放當前媒體，並切換至下一首。

### MusicPlayer 類別
- 子類別 `MusicPlayer` 繼承自父類別 `Player`。
- 方法：
  - `add_song`: 新增歌曲到播放清單，調用父類別的 `add_media` 方法。
  - `remove_song`: 從播放清單中刪除歌曲，調用父類別的 `remove_media` 方法。
  - `play`: 播放音樂，調用父類別的 `play` 方法。

### VideoPlayer 類別
- 子類別 `VideoPlayer` 繼承自父類別 `Player`。
- 方法：
  - `add_video`: 新增影片到播放清單，調用父類別的 `add_media` 方法。
  - `remove_video`: 從播放清單中刪除影片，調用父類別的 `remove_media` 方法。
  - `play`: 播放影片，調用父類別的 `play` 方法。

### 程式碼釋例
```python
if __name__ == "__main__":
    # 創建音樂播放器
    music_player = MusicPlayer()
    music_player.add_song("Song 1")
    music_player.add_song("Song 2")
    print(music_player)
    print(music_player.play())
    print(music_player.play())
    music_player.remove_song("Song 1")
    print(music_player)
    
    # 創建影片播放器
    video_player = VideoPlayer()
    video_player.add_video("Video 1")
    video_player.add_video("Video 2")
    print(video_player)
    print(video_player.play())
    print(video_player.play())
    video_player.remove_video("Video 1")
    print(video_player)

