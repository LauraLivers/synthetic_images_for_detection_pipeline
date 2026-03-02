video = '$HOME/Downloads/20260301_c31_front.mkv'

ffmpeg -i '$video' -vf select='between(t,1,5)+between(t,11,15)' -vsync 0 ${HOME}/Downloads/20260301_c31_front_%d.png ## extract specific window as frames
