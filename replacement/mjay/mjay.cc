// A slightly cleaner version of Mockingjay (HPCA 2022) that works with the latest versions of Champsim (Jun 2023) and uses modern C++ capabilities
// Based on the original version (https://github.com/ishanashah/Mockingjay)
// Functionality should not have changed. All rights belong to the original authors, I guess

#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <vector>

#include "cache.h"
#include "ooo_cpu.h"

constexpr uint32_t HISTORY = 8;
constexpr uint32_t GRANULARITY = 8;
constexpr uint32_t SAMPLED_CACHE_WAYS = 5;
constexpr uint32_t LOG2_SAMPLED_CACHE_SETS = 4;
constexpr uint32_t TIMESTAMP_BITS = 8;

constexpr double TEMP_DIFFERENCE = 1.0 / 16.0;
constexpr double FLEXMIN_PENALTY = 2.0 - lg2(NUM_CPUS) / 4.0;

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
  Counter() = default;

  Counter(const Counter& other) { num = other.num; }

  Counter& operator++()
  {
    num = (num + 1) & bitmask(bits);
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
  uint64_t tag;
  uint64_t signature;
  Timestamp timestamp;

  void reinit(uint64_t pc, uint64_t tag, Timestamp curr)
  {
    valid = true;
    signature = pc;
    this->tag = tag;
    timestamp = curr;
  }
};

class MJData
{
  const uint32_t NUM_SET, NUM_WAY;
  const uint32_t LOG2_SETS, LOG2_SIZE, LOG2_SAMPLED_SETS;

  const uint32_t INF_RD, INF_ETR, MAX_RD;
  const uint32_t SAMPLED_CACHE_TAG_BITS, PC_SIGNATURE_BITS;

  std::vector<std::vector<int32_t>> etr;
  std::vector<uint32_t> etr_clock;
  std::vector<Timestamp> current_timestamp;

  std::unordered_map<uint32_t, uint32_t> rdp;
  std::unordered_map<uint32_t, std::vector<SampledCacheLine>> sampled_cache;

  bool is_sampled_set(uint32_t set)
  {
    uint32_t mask_length = LOG2_SETS - LOG2_SAMPLED_SETS;
    uint32_t mask = bitmask(mask_length);
    return (set & mask) == ((set >> (LOG2_SETS - mask_length)) & mask);
  }

  uint64_t get_pc_signature(uint64_t pc, bool hit, bool prefetch, uint32_t core)
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
      pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
    } else {
      pc = pc << 1;
      if (prefetch) {
        pc = pc | 1;
      }
      pc = pc << 2;
      pc = pc | core;
      pc = CRC_HASH(pc);
      pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
    }
    return pc;
  }

  uint32_t get_sampled_cache_index(uint64_t full_addr)
  {
    full_addr = full_addr >> LOG2_BLOCK_SIZE;
    full_addr = (full_addr << (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_SETS))) >> (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_SETS));
    return full_addr;
  }

  uint64_t get_sampled_cache_tag(uint64_t x)
  {
    x >>= LOG2_SETS + LOG2_BLOCK_SIZE + LOG2_SAMPLED_CACHE_SETS;
    x = (x << (64 - SAMPLED_CACHE_TAG_BITS)) >> (64 - SAMPLED_CACHE_TAG_BITS);
    return x;
  }

  std::optional<uint32_t> search_sampled_cache(uint64_t blockAddress, uint32_t set)
  {
    auto it =
        std::find_if(sampled_cache[set].begin(), sampled_cache[set].end(), [&](SampledCacheLine& entry) { return (entry.valid && entry.tag == blockAddress); });
    if (it != sampled_cache[set].end())
      return std::distance(sampled_cache[set].begin(), it);
    return {};
  }

  void detrain(uint32_t set, uint32_t way)
  {
    SampledCacheLine& temp = sampled_cache[set][way];
    if (!temp.valid)
      return;
    temp.valid = false;

    auto [rdp_it, inserted] = rdp.emplace(temp.signature, INF_RD);
    if (!inserted)
      rdp_it->second = min(rdp_it->second + 1, INF_RD);
  }

  uint32_t temporal_difference(uint32_t init, uint32_t sample)
  {
    if (sample > init) {
      uint32_t diff = sample - init;
      diff = diff * TEMP_DIFFERENCE;
      diff = min(1u, diff);
      return min(init + diff, INF_RD);
    }

    if (sample < init) {
      uint32_t diff = init - sample;
      diff = diff * TEMP_DIFFERENCE;
      diff = min(1u, diff);
      return init - diff;
    }

    return init;
  }

public:
  MJData(uint32_t num_set, uint32_t num_way)
      : NUM_SET{num_set}, NUM_WAY{num_way}, LOG2_SETS{lg2(num_set)}, LOG2_SIZE{LOG2_SETS + lg2(num_way) + LOG2_BLOCK_SIZE},
        LOG2_SAMPLED_SETS{LOG2_SIZE - 16}, INF_RD{NUM_WAY * HISTORY - 1}, INF_ETR{(NUM_WAY * HISTORY / GRANULARITY) - 1}, MAX_RD{INF_RD - 22},
        SAMPLED_CACHE_TAG_BITS{31 - LOG2_SIZE}, PC_SIGNATURE_BITS{LOG2_SIZE - 10}, etr(NUM_SET), etr_clock(NUM_SET), current_timestamp(NUM_SET)
  {
    uint32_t modifier = 1 << LOG2_SETS;
    uint32_t limit = 1 << LOG2_SAMPLED_CACHE_SETS;

    for (uint32_t set = 0; set < NUM_SET; ++set) {
      etr[set].resize(NUM_WAY);
      etr_clock[set] = GRANULARITY;
      if (is_sampled_set(set))
        for (uint32_t i = 0; i < limit; i++)
          sampled_cache.emplace(set + modifier * i, SAMPLED_CACHE_WAYS);
    }
  }

  uint32_t find_victim(uint32_t cpu, uint32_t set, uint64_t pc, uint32_t type)
  {
    auto victim_it = std::min_element(etr[set].begin(), etr[set].end(),
                                      [](int32_t num1, int32_t num2) { return (abs(num1) > abs(num2) || (abs(num1) == abs(num2) && num1 < 0)); });
    uint32_t victim_way = 0;
    uint32_t max_etr = 0;
    if (victim_it != etr[set].end()) {
      victim_way = std::distance(etr[set].begin(), victim_it);
      max_etr = abs(etr[set][victim_way]);
    }

    if (type == WRITEBACK)
      return victim_way;

    auto rdp_it = rdp.find(get_pc_signature(pc, false, type == PREFETCH, cpu));
    if (rdp_it != rdp.end() && (rdp_it->second > MAX_RD || rdp_it->second / GRANULARITY > max_etr))
      return NUM_WAY;

    return victim_way;
  }

  void update_replacement_state(uint32_t cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t pc, uint64_t victim_addr, uint32_t type, uint8_t hit)
  {
    if (type == WRITEBACK) {
      if (!hit)
        etr[set][way] = -INF_ETR;
      return;
    }

    pc = get_pc_signature(pc, hit, type == PREFETCH, cpu);

    if (is_sampled_set(set)) {
      uint32_t sampled_cache_index = get_sampled_cache_index(full_addr);
      uint64_t sampled_cache_tag = get_sampled_cache_tag(full_addr);
      std::optional<uint32_t> sampled_cache_way = search_sampled_cache(sampled_cache_tag, sampled_cache_index);

      auto& sampler_set = sampled_cache[sampled_cache_index];
      if (sampled_cache_way) {
        SampledCacheLine& sampler_entry = sampler_set[sampled_cache_way.value()];
        uint32_t sample = current_timestamp[set] - sampler_entry.timestamp;

        if (sample <= INF_RD) {
          if (type == PREFETCH)
            sample *= FLEXMIN_PENALTY;

          // Either create a new rdp entry for the sample or
          //   get an iterator to the existing one and update it with the sample
          auto [rdp_it, inserted] = rdp.emplace(sampler_entry.signature, sample);
          if (!inserted)
            rdp_it->second = temporal_difference(rdp_it->second, sample);

          sampler_entry.valid = false;
        }
      }

      int lru_way = -1;
      int lru_rd = -1;
      for (uint32_t w = 0; w < SAMPLED_CACHE_WAYS; ++w) {
        if (!sampler_set[w].valid) {
          lru_way = w;
          lru_rd = INF_RD + 1;
          continue;
        }

        uint32_t sample = current_timestamp[set] - sampler_set[w].timestamp;
        if (sample > INF_RD) {
          lru_way = w;
          lru_rd = INF_RD + 1;
          detrain(sampled_cache_index, w);
        } else if (lru_rd < 0 || sample > uint32_t(lru_rd)) {
          lru_way = w;
          lru_rd = sample;
        }
      }
      detrain(sampled_cache_index, lru_way);

      for (auto& entry : sampler_set) {
        if (!entry.valid) {
          entry.reinit(pc, sampled_cache_tag, current_timestamp[set]);
          break;
        }
      }
      ++current_timestamp[set];
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
      etr[set][way] = (rdp[pc] > MAX_RD) ? INF_ETR : rdp[pc] / GRANULARITY;
  }
};

std::unordered_map<CACHE*, MJData> mjay;

/* initialize cache replacement state */
void CACHE::initialize_replacement()
{
  // put your own initialization code here
  mjay.emplace(this, MJData(NUM_SET, NUM_WAY));
}

/* find a cache block to evict
 * return value should be 0 ~ 15 (corresponds to # of ways in cache)
 * current_set: an array of BLOCK, of size 16 */
uint32_t CACHE::find_victim(uint32_t cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t pc, uint64_t full_addr, uint32_t type)
{
  // your eviction policy goes here
  return mjay.at(this).find_victim(cpu, set, pc, type);
}

/* called on every cache hit and cache fill */
void CACHE::update_replacement_state(uint32_t cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t pc, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{
  mjay.at(this).update_replacement_state(cpu, set, way, full_addr, pc, victim_addr, type, hit);
}

/* called at the end of the simulation */
void CACHE::replacement_final_stats() {}
