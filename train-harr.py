!pip install opencv-python

opencv_createsamples -info /content/data/positives.txt -num 1200 -w 64 -h 64 -vec vec_file.vec

opencv_traincascade -data /content/data -vec /content/vec_file.vec -bg /content/data/negatives.txt -numPos 900 -numNeg 500 -numStages 10 -w 24 -h 24

