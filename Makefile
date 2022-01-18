DIRS = src tcpabstract cubic QoSCC vegas reno core app
TARGETS = all clean install



$(TARGETS): %: $(patsubst %, %.%, $(DIRS))




$(foreach TGT, $(TARGETS), $(patsubst %, %.$(TGT), $(DIRS))):
	$(MAKE) -C $(subst ., , $@)
