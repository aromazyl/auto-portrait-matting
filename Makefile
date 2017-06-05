#
# Makefile
# zhangyule, 2017-06-02 16:20
#
#

CC=g++

FLAGS=-lopencv_core -lopencv_imgproc -lopencv_flann -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_objdetect -lopencv_photo -lopencv_features2d -lopencv_calib3d -lopencv_stitching -lopencv_videostab -lopencv_shape -lgflags -lgtest -std=c++11 -g -I /usr/local/include/ -L /usr/local/lib/ -DGTEST -lpthread

all: hog_grabcut test grabcut_ex
	@echo "test all"

people: ./peopledetect.cpp
	${CC} -o peopledetect peopledetect.cpp ${FLAGS}

test: test.cc hog_grabcut_test.cc hog_grabcut gmm_test.cc multi_rect_grab_cut
	${CC} -o test test.cc hog_grabcut_test.cc hog_grabcut.o gmm_test.cc ./multi_rect_grabcut.o ${FLAGS}

hog_grabcut: hog_grabcut.cc hog_grabcut.h multi_rect_grab_cut
	${CC} -c hog_grabcut.cc multi_rect_grabcut.o ${FLAGS}

multi_rect_grab_cut:
	${CC} -c ./multi_rect_grabcut.cc ${FLAGS}

grabcut_ex: ./grab_cut_ex.cpp
	${CC} -o grabcut_ex ./grab_cut_ex.cpp ${FLAGS}

clean:
	rm -rf *.o peopledetect test grabcut_ex core*

