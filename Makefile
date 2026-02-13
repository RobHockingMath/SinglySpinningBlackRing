# One-directory Makefile
# Files expected in THIS folder:
#   render_singly_rotating.cpp
#   singly_rotating_black_ring.cpp
#   singly_rotating_black_ring.h
#   vec_mat.h
#   rk4_adaptive.h
#
# Usage:
#   make -j
#   make run
#   make clean
#
# Options:
#   make DEBUG=1
#   make OMP=0

CXX   ?= g++
DEBUG ?= 0
OMP   ?= 1

BUILD_DIR := build
OBJDIR    := $(BUILD_DIR)/obj
BINDIR    := $(BUILD_DIR)/bin

TARGET    := $(BINDIR)/render_srb.exe

APP_SRC   := render_singly_rotating.cpp
SRB_SRC   := singly_rotating_black_ring.cpp

APP_OBJ   := $(OBJDIR)/render_singly_rotating.o
SRB_OBJ   := $(OBJDIR)/singly_rotating_black_ring.o
OBJS      := $(APP_OBJ) $(SRB_OBJ)

CXXFLAGS_BASE := -std=c++17 -I. -Wall -Wextra -Wpedantic
LDFLAGS_BASE  :=

ifeq ($(DEBUG),1)
  CXXFLAGS_OPT := -O0 -g
else
  CXXFLAGS_OPT := -O3 -DNDEBUG
endif

ifeq ($(OMP),1)
  CXXFLAGS_OMP := -fopenmp -DSRB_USE_OPENMP=1
  LDFLAGS_OMP  := -fopenmp
else
  CXXFLAGS_OMP :=
  LDFLAGS_OMP  :=
endif

CXXFLAGS := $(CXXFLAGS_BASE) $(CXXFLAGS_OPT) $(CXXFLAGS_OMP)
LDFLAGS  := $(LDFLAGS_BASE)  $(LDFLAGS_OMP)

.PHONY: all clean run dirs

all: $(TARGET)

dirs:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(TARGET): dirs $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(APP_OBJ): $(APP_SRC) singly_rotating_black_ring.h vec_mat.h rk4_adaptive.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRB_OBJ): $(SRB_SRC) singly_rotating_black_ring.h vec_mat.h rk4_adaptive.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	@rm -rf $(BUILD_DIR)
