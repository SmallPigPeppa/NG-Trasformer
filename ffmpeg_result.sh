ffmpeg -framerate 10 -i step_%03d.png -vf "scale=800:-1:flags=lanczos" output.gif
