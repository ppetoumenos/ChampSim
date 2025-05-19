#ifndef REPLACEMENT_LRURD_H
#define REPLACEMENT_LRURD_H

#include <unordered_map>
#include <vector>

#include "cache.h"
#include "modules.h"

class RDHistogram {
  public:
    RDHistogram() = delete;
    RDHistogram(uint64_t granularity, uint64_t scale, uint64_t max_distance);
    void increment(champsim::address ip, size_t distance);
    std::vector<uint64_t>& get();
    uint64_t average();
    // What other metrics might be relevant?

  private:
    uint64_t granularity;
    uint64_t log_gran;
    uint64_t scale;
    uint64_t max_distance;
    uint64_t max_bucket;
    std::vector<uint64_t> hist;
    std::unordered_map<uint64_t, std::vector<uint64_t>> ip_hist;

    uint64_t _quantize(uint64_t distance);
    constexpr uint64_t quantize(uint64_t distance);
    constexpr uint64_t unquantize(uint64_t bucket);
};

struct SamplerEntry {
  champsim::address ip;
  uint64_t timestamp;
  uint64_t set_timestamp;
};

class lruRD : public champsim::modules::replacement
{
  long NUM_WAY;
  std::vector<uint64_t> last_used_cycles;
  uint64_t cycle = 0;
  uint64_t access_count = 0;
  std::vector<uint64_t> set_access_count;

  std::unordered_map<uint64_t, SamplerEntry> sampler;

  RDHistogram histogram;
  RDHistogram setrd_histogram;
  std::unordered_map<uint64_t, RDHistogram> ip_histogram;
  std::unordered_map<uint64_t, RDHistogram> setrd_ip_histogram;


public:
  explicit lruRD(CACHE* cache);
  lruRD(CACHE* cache, long sets, long ways);

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                   champsim::address full_addr, access_type type);
  void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                              access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                access_type type, uint8_t hit);
};
#endif
