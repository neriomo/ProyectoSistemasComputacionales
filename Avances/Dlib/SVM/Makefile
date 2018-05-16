
#Incluir direccion de la libreria
DLIB = /home/neriomoran/dlib-19.10

# Direccion del archivo source.cpp
DLIBSRC = $(DLIB)/dlib/all/source.cpp

#Compilador
CXX = g++


OPT = -O3

WARN = -Wall -Wextra -Wcast-align -Wno-sign-compare -Wno-write-strings \
       -Wno-parentheses 

FLAGS = -std=c++11 $(OPTIMIZATION) #$(WARN)

INCLUDE = -I $(DLIB) 

LIBS = -lpthread -lX11 

svm_demo: multiclass_classification_ex.cpp 
	$(CXX) $(FLAGS) $(INCLUDE) $(DLIBSRC) $(LIBS) $^ -o $@

clean: 
	rm -f *.out

ddd:
	valgrind ./multiclass_classification_ex


	

