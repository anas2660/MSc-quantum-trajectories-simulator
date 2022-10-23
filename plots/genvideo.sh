rustc rename.rs && ./rename
ffmpeg -framerate 6 -pattern_type glob -i 'frame*.png' combined_highdt.mp4
