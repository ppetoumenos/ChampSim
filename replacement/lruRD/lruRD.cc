#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

#include "cache.h"

constexpr uint64_t sampler_period{1 << 14};
constexpr uint64_t max_distance{(1 << 27) - 1};

class RDHistogram {
  public:
    RDHistogram() = delete;
    RDHistogram(uint64_t granularity, uint64_t scale, uint64_t max_distance) :
      granularity{granularity},
      log_gran{champsim::msl::lg2(granularity)},
      scale{scale},
      max_distance{max_distance},
      hist(max_bucket + 1) {
        max_bucket = _quantize(max_distance);
      };

    void increment(uint64_t ip, size_t distance) {
      ++hist[quantize(distance)];
	  auto [hist_it, _] = ip_hist.try_emplace(ip, max_bucket + 1);
	  ++hist_it->second[quantize(distance)];
    }

    std::vector<uint64_t>& get() {
      return hist;
    }

    uint64_t average() {
      uint64_t sum = 0;
      uint64_t count = 0;
      for (int i = 0; i <= max_bucket; ++i) {
        count += hist[i];
        sum += hist[i] * unquantize(i);
      }
      return sum / count;
    }
    // What other metrics might be relevant?

  private:
    uint64_t granularity;
    uint64_t log_gran;
    uint64_t scale;
    uint64_t max_distance;
    uint64_t max_bucket;
    std::vector<uint64_t> hist;
	std::unordered_map<uint64_t, std::vector<uint64_t>> ip_hist;

    uint64_t _quantize(uint64_t distance) {
      distance /= scale;

      if (distance < 2 * granularity)
        return distance;

      uint64_t shift = champsim::msl::lg2(distance) - log_gran;
      uint64_t bucket = (distance >> shift) + (shift << log_gran);
      return bucket;
    }

    constexpr uint64_t quantize(uint64_t distance) {
      if (distance > max_distance)
        return max_bucket;
      return _quantize(distance);
    }

    constexpr uint64_t unquantize(uint64_t bucket) {
      if (bucket < 2 * granularity)
        return bucket;

      uint64_t shift = bucket >> log_gran;
      uint64_t distance = ((1ULL << (shift + log_gran - 1)) + (1 << (shift - 1)) * (bucket & (granularity - 1)));
      return distance * scale;
    }
};




struct SamplerEntry {
  uint64_t ip;
  uint64_t timestamp;
  uint64_t set_timestamp;
};


namespace
{
std::map<CACHE*, std::vector<uint64_t>> last_used_cycles;

std::map<CACHE*, uint64_t> access_count;
std::map<CACHE*, std::vector<uint64_t>> set_access_count;
std::map<CACHE*, std::unordered_map<uint64_t, SamplerEntry>> sampler;
std::map<CACHE*, RDHistogram> histogram;
std::map<CACHE*, std::unordered_map<uint64_t, RDHistogram>> ip_histogram;

std::map<CACHE*, RDHistogram> setrd_histogram;
std::map<CACHE*, std::unordered_map<uint64_t, RDHistogram>> setrd_ip_histogram;
}

void CACHE::initialize_replacement() {
  ::last_used_cycles[this] = std::vector<uint64_t>(NUM_SET * NUM_WAY);
  ::access_count[this] = 0;
  ::set_access_count[this] = std::vector<uint64_t>(NUM_SET);
  ::sampler[this].emplace();
  ::histogram.try_emplace(this, 8 * NUM_WAY, NUM_SET / 4, max_distance);
  ::setrd_histogram.try_emplace(this, 2 * NUM_WAY, 1, max_distance / NUM_SET);
}

uint32_t CACHE::find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type)
{
  auto begin = std::next(std::begin(::last_used_cycles[this]), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  // Find the way whose last use cycle is most distant
  auto victim = std::min_element(begin, end);
  assert(begin <= victim);
  assert(victim < end);
  return static_cast<uint32_t>(std::distance(begin, victim)); // cast protected by prior asserts
}

void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{
  // Mark the way as being used on the current cycle
  if (!hit || access_type{type} != access_type::WRITE) // Skip this for writeback hits
    ::last_used_cycles[this].at(set * NUM_WAY + way) = current_cycle;

  if (access_type{type} == access_type::WRITE)
	  return;

  ++access_count[this];
  ++set_access_count[this][set];

  auto it = sampler[this].find(full_addr);
  if (it != sampler[this].end()) {
    uint64_t sample_ip = it->second.ip;
    uint64_t distance = access_count[this] - it->second.timestamp - 1;
    uint64_t set_distance = set_access_count[this][set] - it->second.set_timestamp - 1;
    sampler[this].erase(it);

    histogram.at(this).increment(sample_ip, distance);
    setrd_histogram.at(this).increment(sample_ip, set_distance);
  }

  if (access_count[this] % sampler_period == 0)
    sampler[this][full_addr] = {ip, access_count[this], set_access_count[this][set]};
}

void CACHE::replacement_final_stats(){ }
