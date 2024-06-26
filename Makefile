MDIR := $(shell pwd)

CC = gcc
LIB_LINK = -lgsl -lgslcblas -lm -lfftw3 -lflint
FLAGS = -fPIC -Wall -O3 -ffast-math
FLAGS_DEBUG = -fPIC -Wall -ffast-math -g -O0
BUILD_DIR = $(MDIR)/build
LIB = libfftlogx
SRC_DIR = $(MDIR)/src
_SRC = \
	cfftlog.c \
	utils.c \
	utils_complex.c \

SRC = $(addprefix $(SRC_DIR)/,$(_SRC))


all: $(LIB).so

debug: FLAGS = $(FLAGS_DEBUG)
debug: $(LIB).so

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB).so: $(BUILD_DIR)
	$(CC) -shared -o $(BUILD_DIR)/$(LIB).so $(SRC) $(LIB_LINK) $(FLAGS)

clean:
	rm $(BUILD_DIR)/*.so