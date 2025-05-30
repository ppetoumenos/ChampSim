// A slightly cleaner version of Mockingjay (HPCA 2022) that works with the latest versions of Champsim (Jun 2023) and uses modern C++ capabilities
// Based on the original version (https://github.com/ishanashah/Mockingjay)
// Functionality should not have changed. All rights belong to the original authors, I guess

#include <optional>
#include <unordered_map>
#include <vector>

#include "cache.h"
#include "modules.h"

//#define PRODUCE_RD_TRACE

constexpr uint32_t ADDR_BITS = 64;

constexpr uint32_t HISTORY = 8;
constexpr uint32_t GRANULARITY = 8;
constexpr uint32_t SAMPLED_CACHE_WAYS = 5;
constexpr uint32_t LOG2_SAMPLED_CACHE_SETS = 4;
constexpr uint32_t TIMESTAMP_BITS = 8;

constexpr uint32_t TEMPDIFF_SCALING = 16;


#ifdef PRODUCE_RD_TRACE
struct TraceSamplerEntry {
  uint64_t ip;
  uint32_t timestamp;
  uint32_t set_timestamp;
};

// Asssume that 32 bits are enough for full cache rds
// And 16 bits are enough for set rds
struct TraceData {
  uint64_t ip;
  uint64_t addr;
  uint32_t past_rd;
  uint32_t future_rd;
  uint16_t past_rd_set;
  uint16_t future_rd_set;

  TraceData(uint64_t _ip, uint64_t _addr, uint32_t _past_rd, uint32_t _future_rd, uint16_t _past_rd_set, uint16_t _future_rd_set) :
   ip{_ip}, addr{_addr}, past_rd{_past_rd}, future_rd{_future_rd}, past_rd_set{_past_rd_set}, future_rd_set{_future_rd_set}   {}
};
#endif


template <uint32_t bits>
class Counter
{
  uint32_t num{0};

public:
  Counter& operator++()
  {
    num = (num + 1) & champsim::bitmask(champsim::data::bits{bits});
    return *this;
  }

  uint32_t operator-(const Counter& other)
  {
    uint32_t val = num;
    if (val < other.num)
      val += (1 << bits);
    return val - other.num;
  }
};

using Timestamp = Counter<TIMESTAMP_BITS>;

struct SampledCacheLine {
  bool valid{false};
  uint64_t tag{0};
  uint64_t signature{0};
  Timestamp timestamp{};

  void set(uint64_t pc, uint64_t _tag, Timestamp curr)
  {
    valid = true;
    signature = pc;
    this->tag = _tag;
    timestamp = curr;
  }
};

class Sampler
{
  const uint32_t LOG2_SETS, LOG2_SIZE, LOG2_SAMPLED_SETS;
  const uint32_t SAMPLED_CACHE_TAG_BITS;
  const uint32_t INF_RD;

  std::unordered_map<uint32_t, std::vector<SampledCacheLine>> samples;
  std::vector<Timestamp> current_timestamp;

  uint32_t get_index(uint64_t full_addr) const;
  uint64_t get_tag(uint64_t full_addr) const;

public:
  Sampler(uint32_t num_cache_set, uint32_t num_cache_way);

  bool is_sampled(uint32_t cache_set) const;
  std::optional<std::pair<uint64_t, uint32_t>> get_sample(uint32_t cache_set, uint64_t full_addr);
  std::vector<uint64_t> add_sample(uint32_t cache_set, uint64_t full_addr, uint64_t pc);
};

class MJ : public champsim::modules::replacement
{
  const uint32_t NUM_SET, NUM_WAY, LOG2_SIZE;

  const uint32_t INF_RD, MAX_RD;
  const int32_t INF_ETR;
  const uint32_t PC_SIGNATURE_BITS;
  const double FLEXMIN_PENALTY = 2.0 - static_cast<double>(champsim::msl::lg2(NUM_CPUS)) / 4.0;

  std::vector<std::vector<int32_t>> etr;
  std::vector<uint32_t> etr_clock;

  std::unordered_map<uint32_t, uint32_t> rdp;
  Sampler sampler;

#ifdef PRODUCE_RD_TRACE
  uint32_t access_count{0};
  std::vector<uint32_t> set_access_count;
  std::unordered_map<uint64_t, TraceSamplerEntry> traceSampler;
  std::vector<TraceData> trace;
#endif

  uint64_t get_pc_signature(champsim::address ip, bool hit, bool prefetch, uint32_t core) const;
  uint32_t temporal_difference(uint32_t init, uint32_t sample) const;

public:
  explicit MJ(CACHE* cache) : MJ(cache, cache->NUM_SET, cache->NUM_WAY) {};
  MJ(CACHE* cache, uint32_t num_set, uint32_t num_way);

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip, champsim::address full_addr, access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr, access_type type, uint8_t hit);
  void replacement_final_stats();
};


