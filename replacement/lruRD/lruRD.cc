#include <algorithm>
#include <cassert>

#include "lruRD.h"

constexpr uint64_t sampler_period{1 << 14};
constexpr uint64_t max_distance{(1 << 27) - 1};

RDHistogram::RDHistogram(uint64_t _granularity, uint64_t _scale, uint64_t _max_distance) :
  granularity{_granularity},
  log_gran{champsim::msl::lg2(_granularity)},
  scale{_scale},
  max_distance{_max_distance}
{
  max_bucket = _quantize(max_distance);
  hist.resize(max_bucket + 1, 0);
};

void RDHistogram::increment(champsim::address ip, size_t distance) {
  ++hist[quantize(distance)];
  auto [hist_it, _] = ip_hist.try_emplace(ip.to<uint64_t>(), max_bucket + 1);
  ++hist_it->second[quantize(distance)];
}

std::vector<uint64_t>& RDHistogram::get() {
  return hist;
}

uint64_t RDHistogram::average() {
  uint64_t sum = 0;
  uint64_t count = 0;
  for (unsigned i = 0; i <= max_bucket; ++i) {
    count += hist[i];
    sum += hist[i] * unquantize(i);
  }
  return sum / count;
}

uint64_t RDHistogram::_quantize(uint64_t distance) {
  distance /= scale;

  if (distance < 2 * granularity)
    return distance;

  uint64_t shift = champsim::msl::lg2(distance) - log_gran;
  uint64_t bucket = (distance >> shift) + (shift << log_gran);
  return bucket;
}

constexpr uint64_t RDHistogram::quantize(uint64_t distance) {
  if (distance > max_distance)
    return max_bucket;
  return _quantize(distance);
}

constexpr uint64_t RDHistogram::unquantize(uint64_t bucket) {
  if (bucket < 2 * granularity)
    return bucket;

  uint64_t shift = bucket >> log_gran;
  uint64_t distance = ((1ULL << (shift + log_gran - 1)) + (1 << (shift - 1)) * (bucket & (granularity - 1)));
  return distance * scale;
}


lruRD::lruRD(CACHE* cache) : lruRD(cache, cache->NUM_SET, cache->NUM_WAY) {}

lruRD::lruRD(CACHE* cache, long sets, long ways) :
  replacement(cache), NUM_WAY(ways),
  last_used_cycles(static_cast<std::size_t>(sets * ways), 0),
  set_access_count(static_cast<std::size_t>(sets * ways), 0),
  sampler{},
  histogram(8 * ways, sets / 4, max_distance),
  setrd_histogram(2 * ways, 1, max_distance / sets) {}

long lruRD::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                  champsim::address full_addr, access_type type)
{
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  // Find the way whose last use cycle is most distant
  auto victim = std::min_element(begin, end);
  assert(begin <= victim);
  assert(victim < end);
  return std::distance(begin, victim);
}

void lruRD::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                             access_type type)
{
  // Mark the way as being used on the current cycle
  last_used_cycles.at((std::size_t)(set * NUM_WAY + way)) = cycle++;
}

void lruRD::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                               champsim::address victim_addr, access_type type, uint8_t hit)
{
  // Mark the way as being used on the current cycle
  if (hit && access_type{type} != access_type::WRITE) // Skip this for writeback hits
  last_used_cycles.at((std::size_t)(set * NUM_WAY + way)) = cycle++;

  if (access_type{type} == access_type::WRITE)
    return;

  ++access_count;
  ++set_access_count[set];

  auto it = sampler.find(full_addr.to<uint64_t>());
  if (it != sampler.end()) {
    champsim::address sample_ip = it->second.ip;
    uint64_t distance = access_count - it->second.timestamp - 1;
    uint64_t set_distance = set_access_count[set] - it->second.set_timestamp - 1;
    sampler.erase(it);

    histogram.increment(sample_ip, distance);
    setrd_histogram.increment(sample_ip, set_distance);
  }

  if (access_count % sampler_period == 0)
    sampler[full_addr.to<uint64_t>()] = {ip, access_count, set_access_count[set]};
}
