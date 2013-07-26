#bin_PROGRAMS = library_example
bin_PROGRAMS = scp
#bin_PROGRAMS = recommend
#library_example_SOURCES = main.cpp environment.cpp item.cpp submodOracle.cpp seqMachine.cpp
scp_SOURCES = main.cpp environment.cpp item.cpp submodOracle.cpp seqMachine.cpp

#library_example_SOURCES = ezexample_predict.cc
#library_example_LDADD = ../vowpalwabbit/libvw.la ../vowpalwabbit/liballreduce.la

scp_LDADD = ../vowpalwabbit/libvw.la ../vowpalwabbit/liballreduce.la

ACLOCAL_AMFLAGS = -I acinclude.d

INCLUDES = ${BOOST_CPPFLAGS} ${ZLIB_CPPFLAGS} ${PTHREAD_CFLAGS}
AM_LDFLAGS = ${BOOST_LDFLAGS} ${BOOST_PROGRAM_OPTIONS_LIB} ${ZLIB_LDFLAGS} ${PTHREAD_LIBS}
