C++ = g++

ifndef os
   os = LINUX
endif

ifndef arch
   arch = IA32
endif

#CCFLAGS = -Wall -D$(os) -I../core -I/home/qqcai/anaconda3/include/python3.8/ -finline-functions -O3
#CCFLAGS = -Wall -D$(os) -I/home/QoSCC/core -I/home/python3.7/include/python3.7m -finline-functions -O3
#CCFLAGS = -Wall -D$(os) -I../core -I/home/cqq/anaconda3/include/python3.8/ -finline-functions -O3
CCFLAGS = -Wall -D$(os) -I../core -finline-functions -O3
#CCFLAGS = -Wall -D$(os) -I/home/qqcai/anaconda3/include/python3.8/ -L/home/qqcai/anaconda3/lib/ -finline-functions -O3

ifeq ($(arch), IA32)
   CCFLAGS += -DIA32 #-mcpu=pentiumpro -march=pentiumpro -mmmx -msse
endif

ifeq ($(arch), POWERPC)
   CCFLAGS += -mcpu=powerpc
endif

ifeq ($(arch), IA64)
   CCFLAGS += -DIA64
endif

ifeq ($(arch), SPARC)
   CCFLAGS += -DSPARC
endif

#LDFLAGS = -L../core -ludt -lstdc++ -lpthread -lm
LDFLAGS = -L../core -ludt -lstdc++ -lpthread -lm
#LDFLAGS = -L../core -L/home/cqq/anaconda3/envs/py36_tensor/lib/ -ludt -lstdc++ -lpthread -lm -lpython3.6m -fno-lto
#LDFLAGS = -L../core -ludt -lstdc++ -lpthread -lm -L/home/python3.7/lib -lpython3.7m #-fno-lto
#LDFLAGS = -L/home/QoSCC/core -ludt -L/home/python3.7/lib -lpython3.7m  -lpthread -lm -lutil -ldl  # -lstdc++ -fno-lto
#LDFLAGS = -L../core -L/home/cqq/anaconda3/lib/ -ludt -lstdc++ -lpthread -lm -fno-lto

ifeq ($(os), UNIX)
   LDFLAGS += -lsocket
endif

ifeq ($(os), SUNOS)
   LDFLAGS += -lrt -lsocket
endif

DIR = $(shell pwd)

APP = appserver appclient

all: $(APP)

%.o: %.cpp
	$(C++) $(CCFLAGS) $< -c

appserver: appserver.o
	$(C++) $^ -o $@ $(LDFLAGS) 
appclient: appclient.o
	$(C++) $^ -o $@ $(LDFLAGS) 
 

clean:
	rm -f *.o $(APP)

install:
	export PATH=$(DIR):$$PATH
