set(BENCHMARK_HEADERS
  file.h
)

set(BENCHMARK_SOURCES
)

ADD_FASTER_BENCHMARK(benchmark)
ADD_FASTER_BENCHMARK(recover_benchmark)
ADD_FASTER_BENCHMARK(variable_length_benchmark)

add_executable(process_ycsb process_ycsb.cc)
