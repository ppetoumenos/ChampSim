// A slightly cleaner version of Mockingjay (HPCA 2022) that works with the latest versions of Champsim (Jun 2023) and uses modern C++ capabilities
// Based on the original version (https://github.com/ishanashah/Mockingjay)
// Functionality should not have changed. All rights belong to the original authors, I guess

#include <cstdlib>
#include <optional>
#include <vector>

#include "cache.h"
#include "ooo_cpu.h"
#include "msl/bits.h"
#include <unordered_map>

#define PRODUCE_RD_TRACE
#ifdef PRODUCE_RD_TRACE
#include <fstream>
#endif

constexpr uint32_t ADDR_BITS = 64;

constexpr uint32_t HISTORY = 8;
constexpr uint32_t GRANULARITY = 8;
constexpr uint32_t SAMPLED_CACHE_WAYS = 5;
constexpr uint32_t LOG2_SAMPLED_CACHE_SETS = 4;
constexpr uint32_t TIMESTAMP_BITS = 8;

constexpr uint32_t TEMPDIFF_SCALING = 16;
constexpr double FLEXMIN_PENALTY = 2.0 - champsim::msl::lg2(NUM_CPUS) / 4.0;


#ifdef PRODUCE_RD_TRACE
struct SamplerEntry {
  uint64_t ip;
  uint32_t timestamp;
  uint32_t set_timestamp;
};
#endif


uint64_t CRC_HASH(uint64_t _blockAddress)
{
  static const unsigned long long crcPolynomial = 3988292384ULL;
  uint64_t _returnVal = _blockAddress;
  _returnVal = ((_returnVal & 1) == 1) ? ((_returnVal >> 1) ^ crcPolynomial) : (_returnVal >> 1);
  _returnVal = ((_returnVal & 1) == 1) ? ((_returnVal >> 1) ^ crcPolynomial) : (_returnVal >> 1);
  _returnVal = ((_returnVal & 1) == 1) ? ((_returnVal >> 1) ^ crcPolynomial) : (_returnVal >> 1);
  return _returnVal;
}

template <uint32_t bits>
class Counter
{
  uint32_t num{0};

public:
  Counter& operator++()
  {
    num = (num + 1) & champsim::msl::bitmask(bits);
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

  uint32_t get_index(uint64_t full_addr) const
  {
    full_addr = full_addr >> LOG2_BLOCK_SIZE;
    full_addr = (full_addr << (ADDR_BITS - (LOG2_SAMPLED_CACHE_SETS + LOG2_SETS))) >> (ADDR_BITS - (LOG2_SAMPLED_CACHE_SETS + LOG2_SETS));
    return full_addr;
  }

  uint64_t get_tag(uint64_t full_addr) const
  {
    full_addr >>= LOG2_SETS + LOG2_BLOCK_SIZE + LOG2_SAMPLED_CACHE_SETS;
    full_addr = (full_addr << (ADDR_BITS - SAMPLED_CACHE_TAG_BITS)) >> (ADDR_BITS - SAMPLED_CACHE_TAG_BITS);
    return full_addr;
  }

public:
  Sampler(uint32_t num_cache_set, uint32_t num_cache_way)
      : LOG2_SETS{champsim::msl::lg2(num_cache_set)}, LOG2_SIZE{LOG2_SETS + champsim::msl::lg2(num_cache_way) + LOG2_BLOCK_SIZE}, LOG2_SAMPLED_SETS{LOG2_SIZE - 16},
        SAMPLED_CACHE_TAG_BITS{31 - LOG2_SIZE}, INF_RD{num_cache_way * HISTORY - 1}, current_timestamp(num_cache_set)
  {
    uint32_t modifier = 1 << LOG2_SETS;
    uint32_t limit = 1 << LOG2_SAMPLED_CACHE_SETS;

    for (uint32_t set = 0; set < num_cache_set; ++set)
      if (is_sampled(set))
        for (uint32_t i = 0; i < limit; i++)
          samples.emplace(set + modifier * i, SAMPLED_CACHE_WAYS);
  }

  bool is_sampled(uint32_t cache_set) const
  {
    uint32_t mask_length = LOG2_SETS - LOG2_SAMPLED_SETS;
    uint32_t mask = champsim::msl::bitmask(mask_length);
    return (cache_set & mask) == ((cache_set >> (LOG2_SETS - mask_length)) & mask);
  }

  std::optional<std::pair<uint64_t, uint32_t>> get_sample(uint32_t cache_set, uint64_t full_addr)
  {
    uint64_t tag = get_tag(full_addr);
    auto& set = samples.at(get_index(full_addr));
    auto sample_it = std::find_if(set.begin(), set.end(), [&](SampledCacheLine& sample) { return (sample.valid && sample.tag == tag); });

    if (sample_it == set.end())
      return {};

    uint32_t distance = current_timestamp[cache_set] - sample_it->timestamp;
    if (distance > INF_RD)
      return {};

    sample_it->valid = false;
    return {{sample_it->signature, distance}};
  }

  std::vector<uint64_t> add_sample(uint32_t cache_set, uint64_t full_addr, uint64_t pc)
  {
    std::vector<uint64_t> expired;
    auto& set = samples.at(get_index(full_addr));

    bool found_invalid = false;
    bool found_valid = false;
    uint32_t lru_way = 0;
    uint32_t lru_rd = 0;

    for (uint32_t way = 0; way < set.size(); ++way) {
      SampledCacheLine& sample = set[way];
      if (!sample.valid) {
        found_invalid = true;
        continue;
      }

      uint32_t distance = current_timestamp[cache_set] - sample.timestamp;
      if (distance > INF_RD) {
        expired.push_back(sample.signature);
        sample.valid = false;
        found_invalid = true;
      } else if (!found_valid || distance > lru_rd) {
        lru_way = way;
        lru_rd = distance;
        found_valid = true;
      }
    }

    if (!found_invalid) {
      expired.push_back(set[lru_way].signature);
      set[lru_way].valid = false;
    }

    for (auto& sample : set) {
      if (!sample.valid) {
        sample.set(pc, get_tag(full_addr), current_timestamp[cache_set]);
        break;
      }
    }
    ++current_timestamp[cache_set];
    return expired;
  }
};

class MJData
{
  const uint32_t NUM_SET, NUM_WAY, LOG2_SIZE;

  const uint32_t INF_RD, MAX_RD;
  const int32_t INF_ETR;
  const uint32_t PC_SIGNATURE_BITS;

  std::vector<std::vector<int32_t>> etr;
  std::vector<uint32_t> etr_clock;

  std::unordered_map<uint32_t, uint32_t> rdp;
  Sampler sampler;

  uint64_t get_pc_signature(uint64_t pc, bool hit, bool prefetch, uint32_t core) const
  {
    if (NUM_CPUS == 1) {
      pc = pc << 1;
      if (hit) {
        pc = pc | 1;
      }
      pc = pc << 1;
      if (prefetch) {
        pc = pc | 1;
      }
      pc = CRC_HASH(pc);
      pc = (pc << (ADDR_BITS - PC_SIGNATURE_BITS)) >> (ADDR_BITS - PC_SIGNATURE_BITS);
    } else {
      pc = pc << 1;
      if (prefetch) {
        pc = pc | 1;
      }
      pc = pc << 2;
      pc = pc | core;
      pc = CRC_HASH(pc);
      pc = (pc << (ADDR_BITS - PC_SIGNATURE_BITS)) >> (ADDR_BITS - PC_SIGNATURE_BITS);
    }
    return pc;
  }

  uint32_t temporal_difference(uint32_t init, uint32_t sample)
  {
    if (sample > init) {
      uint32_t diff = sample - init;
      diff /= TEMPDIFF_SCALING;
      diff = std::min(1U, diff);
      return std::min(init + diff, INF_RD);
    }

    if (sample < init) {
      uint32_t diff = init - sample;
      diff /= TEMPDIFF_SCALING;
      diff = std::min(1U, diff);
      return init - diff;
    }

    return init;
  }

public:
  MJData(uint32_t num_set, uint32_t num_way)
      : NUM_SET{num_set}, NUM_WAY{num_way}, LOG2_SIZE{champsim::msl::lg2(num_set) + champsim::msl::lg2(num_way) + LOG2_BLOCK_SIZE}, INF_RD{NUM_WAY * HISTORY - 1}, MAX_RD{INF_RD - 22},
        INF_ETR{static_cast<int32_t>((NUM_WAY * HISTORY / GRANULARITY) - 1)}, PC_SIGNATURE_BITS{LOG2_SIZE - 10}, etr(NUM_SET), etr_clock(NUM_SET),
        sampler(num_set, num_way)
  {
    for (uint32_t set = 0; set < NUM_SET; ++set) {
      etr[set].resize(NUM_WAY);
      etr_clock[set] = GRANULARITY;
    }
  }

  uint32_t find_victim(uint32_t triggering_cpu, uint32_t set, uint64_t pc, uint32_t type)
  {
    auto victim_it = std::min_element(etr[set].begin(), etr[set].end(),
                                      [](int32_t num1, int32_t num2) { return (abs(num1) > abs(num2) || (abs(num1) == abs(num2) && num1 < 0)); });
    uint32_t victim_way = 0;
    uint32_t max_etr = 0;
    if (victim_it != etr[set].end()) {
      victim_way = std::distance(etr[set].begin(), victim_it);
      max_etr = abs(etr[set][victim_way]);
    }

    if (access_type{type} == access_type::WRITE)
      return victim_way;

    auto rdp_it = rdp.find(get_pc_signature(pc, false, access_type{type} == access_type::PREFETCH, triggering_cpu));
    if (rdp_it != rdp.end() && (rdp_it->second > MAX_RD || rdp_it->second / GRANULARITY > max_etr))
      return NUM_WAY;

    return victim_way;
  }

  void update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t pc, uint64_t victim_addr, uint32_t type, uint8_t hit)
  {
    if (access_type{type} == access_type::WRITE) {
      if (!hit)
        etr[set][way] = -INF_ETR;
      return;
    }

    pc = get_pc_signature(pc, hit, access_type{type} == access_type::PREFETCH, triggering_cpu);

    if (sampler.is_sampled(set)) {
      std::optional<std::pair<uint64_t, uint32_t>> sample = sampler.get_sample(set, full_addr);
      if (sample) {
        auto [signature, distance] = sample.value();
        if (access_type{type} == access_type::PREFETCH)
          distance *= FLEXMIN_PENALTY;

        // Either create a new rdp entry for the sample or
        //   get an iterator to the existing one and update it with the sample
        auto [rdp_it, inserted] = rdp.emplace(signature, distance);
        if (!inserted)
          rdp_it->second = temporal_difference(rdp_it->second, distance);
      }

      std::vector<uint64_t> expired_samples = sampler.add_sample(set, full_addr, pc);

      for (uint64_t signature : expired_samples) {
        auto [rdp_it, inserted] = rdp.emplace(signature, INF_RD);
        if (!inserted)
          rdp_it->second = std::min(rdp_it->second + 1, INF_RD);
      }
    }

    if (etr_clock[set] == GRANULARITY) {
      for (uint32_t w = 0; w < NUM_WAY; ++w) {
        if (w != way && abs(etr[set][w]) < INF_ETR) {
          etr[set][w]--;
        }
      }
      etr_clock[set] = 0;
    }
    etr_clock[set]++;

    if (way == NUM_WAY)
      return;

    if (!rdp.count(pc))
      etr[set][way] = (NUM_CPUS == 1) ? 0 : INF_ETR;
    else
      etr[set][way] = (rdp[pc] > MAX_RD) ? INF_ETR : static_cast<int32_t>(rdp[pc] / GRANULARITY);
  }
};

std::unordered_map<CACHE*, MJData> mjay;

#ifdef PRODUCE_RD_TRACE
std::unordered_map<CACHE*, uint32_t> access_count;
std::unordered_map<CACHE*, std::vector<uint32_t>> set_access_count;
std::unordered_map<CACHE*, std::unordered_map<uint64_t, SamplerEntry>> sampler;

// Let's assume we have only one cache using this
// Also assume that 32 bits are enough for full cache rds
// And 16 bits are enough for set rds
struct traceData {
  uint64_t ip;
  uint64_t addr;
  uint32_t past_rd;
  uint32_t future_rd;
  uint16_t past_rd_set;
  uint16_t future_rd_set;

  traceData(uint64_t _ip, uint64_t _addr, uint32_t _past_rd, uint32_t _future_rd, uint16_t _past_rd_set, uint16_t _future_rd_set) :
   ip{_ip}, addr{_addr}, past_rd{_past_rd}, future_rd{_future_rd}, past_rd_set{_past_rd_set}, future_rd_set{_future_rd_set} 	{}
};
std::vector<traceData> trace;
#endif

/* initialize cache replacement state */
void CACHE::initialize_replacement()
{
  // put your own initialization code here
  mjay.emplace(this, MJData(NUM_SET, NUM_WAY));
#ifdef PRODUCE_RD_TRACE
  access_count[this] = 0;
  set_access_count.try_emplace(this, NUM_SET, 0);
  assert(set_access_count[this].size() == NUM_SET);
  //for (int i = 0; i < NUM_SET; ++i)
  //  set_access_count[this].push_back(0);
  sampler.try_emplace(this);
#endif
}

/* find a cache block to evict
 * return value should be 0 ~ 15 (corresponds to # of ways in cache)
 * current_set: an array of BLOCK, of size 16 */
uint32_t CACHE::find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t pc, uint64_t full_addr, uint32_t type)
{
  // your eviction policy goes here
  return mjay.at(this).find_victim(triggering_cpu, set, pc, type);
}

/* called on every cache hit and cache fill */
void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t pc, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{
  mjay.at(this).update_replacement_state(triggering_cpu, set, way, full_addr, pc, victim_addr, type, hit);

#ifdef PRODUCE_RD_TRACE
  if (access_type{type} == access_type::WRITE)
    return;

  ++access_count[this];
  ++set_access_count[this][set];

  auto it = sampler[this].find(full_addr);
  if (it != sampler[this].end()) {
    uint64_t sample_ip = it->second.ip;

    uint32_t distance = access_count[this] - it->second.timestamp - 1;
    uint32_t set_distance = set_access_count[this][set] - it->second.set_timestamp - 1;
	// Clamp to 16 bits
	if (set_distance > std::numeric_limits<uint16_t>::max())
      set_distance = std::numeric_limits<uint16_t>::max();

    sampler[this].erase(it);
    trace.emplace_back(sample_ip, full_addr, static_cast<uint32_t>(distance), std::numeric_limits<uint32_t>::max(), static_cast<uint16_t>(set_distance), std::numeric_limits<uint16_t>::max());
  } else {
    trace.emplace_back(0, full_addr, std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::max());
  }
  sampler[this][full_addr] = {pc, access_count[this], set_access_count[this][set]};
#endif
}

/* called at the end of the simulation */
void CACHE::replacement_final_stats() {
#ifdef PRODUCE_RD_TRACE
  std::unordered_map<uint64_t, std::vector<traceData>::reverse_iterator> last_seen;
  for (auto it = trace.rbegin(); it != trace.rend(); ++it) {
    auto entry = last_seen.find(it->addr);
    if (entry != last_seen.end()) {
      it->future_rd = entry->second->past_rd;
      it->future_rd_set = entry->second->past_rd_set;
    }
    last_seen[it->addr] = it;
  }

  std::ofstream trace_file("trace.csv");
  // Remove as many fields as needed
  for (auto it = trace.begin(); it != trace.end(); ++it) {
    trace_file << it->ip << "\t" << it->addr << "\t" << it->past_rd << "\t" << it->past_rd_set << "\t" << it->future_rd << "\t" << it->future_rd_set << "\n";
  }

#endif
}
