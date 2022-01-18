C++ = g++ -g #-std=c++11

ifndef os
  os = LINUX
endif

ifndef arch
  arch = IA32
endif

#CCFLAGS = -I/home/qqcai/anaconda3/include/python3.8/ -L/home/qqcai/anaconda3/lib/ -fPIC -Wall -Wextra -D$(os) -finline-functions -O3 -fno-strict-aliasing #-msse3
CCFLAGS = -I/home/python3.7/include/python3.7m -L/home/python3.7/lib -lpython3.7m -lpthread -lutil -ldl -fPIC -Wall -Wextra -D$(os) -finline-functions -O3 -fno-strict-aliasing #-msse3

ifeq ($(arch), IA32)
  CCFLAGS += -DIA32
endif

ifeq ($(arch), POWERPC)
  CCFLAGS += -mcpu=powerpc
endif

ifeq ($(arch), SPARC)
  CCFLAGS += -DSPARC
endif

ifeq ($(arch), IA64)
  CCFLAGS += -DIA64
endif

ifeq ($(arch), AMD64)
  CCFLAGS += -DAMD64
endif

OBJS = QoSCC.o
DIR = $(shell pwd) # shell函数

all: $(OBJS)

%.o: %.cpp %.h #../lib/alglib/src/%.cpp
	$(C++) $(CCFLAGS) $< -c

clean:
	rm -f *.o
