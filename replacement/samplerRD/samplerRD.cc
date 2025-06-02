#include <fstream>

#include "samplerRD.h"

void HistVector::increment(uint64_t idx) {
#ifdef SPARSE_HISTOGRAMS
  auto [it, inserted] = counts.try_emplace(idx, 1);
  if (!inserted)
    it->second++;
#else
  ++counts[idx];
#endif
}

auto HistVector::get() const {
#ifdef SPARSE_HISTOGRAMS
  return std::vector<std::pair<uint64_t, uint64_t>>(counts.begin(), counts.end());
#else
  std::vector<std::pair<uint64_t, uint64_t>> result;
  for (int i = 0; i < counts.size(); ++i)
    if (counts[i] > 0)
      result.emplace_back(i, counts[i]);
  return result;
#endif
}

void HistVector::resize(size_t new_size, uint64_t value) {
#ifdef SPARSE_HISTOGRAMS
	return;
#else
	counts.resize(new_size, value);
#endif
}


HistogramRD::HistogramRD(uint64_t _granularity, uint64_t _scale, uint64_t _max_distance) :
  granularity{_granularity}, log_gran{champsim::msl::lg2(_granularity)},
  scale{_scale}, max_distance{_max_distance}
{
  max_bucket = _quantize(max_distance);
  hist.resize(max_bucket + 1, 0);
}

void HistogramRD::increment(champsim::address ip, size_t distance) {
  uint64_t bucket = quantize(distance);
  hist.increment(bucket);
  auto [hist_it, _] = ip_hist.try_emplace(ip.to<uint64_t>());
  hist_it->second.increment(bucket);
}

uint64_t HistogramRD::average() const {
  uint64_t sum = 0;
  uint64_t total_count = 0;
  for (auto [bucket, count]: hist.get()) {
    total_count += count;
    sum += count * unquantize(bucket);
  }
  return sum / total_count;
}


void HistogramRD::print(const std::string& filename) {
  std::ofstream file{filename};

  for (auto [bucket, count]: hist.get()) 
    file << "0\t" << unquantize(bucket) << "\t" << count << "\n";

  for (auto [ip, counts]: ip_hist)
    for (auto [bucket, count]: counts.get())
      file << ip << "\t" << unquantize(bucket) << "\t" << count << "\n";
}

uint64_t HistogramRD::_quantize(uint64_t distance) const {
  distance /= scale;

  if (granularity == 0 || distance < 2 * granularity)
    return distance;

  uint64_t shift = champsim::msl::lg2(distance) - log_gran;
  uint64_t bucket = (distance >> shift) + (shift << log_gran);
  return bucket;
}

constexpr uint64_t HistogramRD::quantize(uint64_t distance) const {
  if (distance > max_distance)
    return max_bucket;
  return _quantize(distance);
}

constexpr uint64_t HistogramRD::unquantize(uint64_t bucket) const {
  if (granularity == 0 || bucket < 2 * granularity)
    return bucket * scale;

  uint64_t shift = bucket >> log_gran;
  uint64_t distance = ((1ULL << (shift + log_gran - 1)) + (1 << (shift - 1)) * (bucket & (granularity - 1)));
  return distance * scale;
}

samplerRD::samplerRD(CACHE* cache) : samplerRD(cache, cache->NUM_SET, cache->NUM_WAY) {}

samplerRD::samplerRD(CACHE* cache, long _sets, long _ways) :
	replacement(cache),
    basename{cache->NAME},
    set_access_count(static_cast<std::size_t>(_sets), 0),
    samples{},
    granularity_full{linear_buckets ? 0 : 8 * static_cast<uint64_t>(_ways)},
    granularity_set{linear_buckets ? 0 : 2 * static_cast<uint64_t>(_ways)},
    //scaling_full{static_cast<uint64_t>(_sets * _ways) / 32},
    scaling_full{scaling},
    scaling_set{1},
    max_distance_full{max_distance},
    max_distance_set{max_distance / scaling},
    ignore_writes{cache->NAME.substr(2) != "L1"} {}

void samplerRD::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr, access_type type, uint8_t hit) {
  // Ignore non-programmatic accessses
  // CHECK WHETHER THIS IS THE RIGHT BEHAVIOR FOR WHAT YOU ARE TRYING TO ACHIEVE
  if (type != access_type::LOAD && type != access_type::WRITE)
    return;
  if (ignore_writes && type == access_type::WRITE)
    return;

  size_t set_idx = static_cast<size_t>(set);

  auto it = samples.find(full_addr.to<uint64_t>());

  if (it != samples.end()) {
    champsim::address sample_ip = it->second.ip;
    uint64_t distance = access_count - it->second.timestamp - 1;
    uint64_t set_distance = set_access_count[set_idx] - it->second.set_timestamp - 1;
    uint32_t window_id = it->second.window_id;
    samples.erase(it);

	while (window_id >= histograms.size()) {
		histograms.emplace_back(granularity_full, scaling_full, max_distance_full);
		set_histograms.emplace_back(granularity_set, scaling_set, max_distance_set);
	}

    histograms[window_id].increment(sample_ip, distance);
    set_histograms[window_id].increment(sample_ip, set_distance);
  }

  if (access_count >= sampling_start && access_count < sampling_end && std::generate_canonical<double, 16>(rng) < sampling_rate)
    samples[full_addr.to<uint64_t>()] = {ip, access_count, set_access_count[set_idx], current_window};

  ++access_count;
  ++set_access_count[set_idx];

  if (access_count == sampling_end) {
    ++current_window;
    uint64_t hib_upper_bound = (hibernation_length * 11) / 10;
    uint64_t hib_lower_bound = (hibernation_length * 9) / 10;
    sampling_start = access_count + std::uniform_int_distribution<uint64_t>(hib_lower_bound, hib_upper_bound)(rng);
    sampling_end = sampling_start + sampling_length;
  }
}

void samplerRD::replacement_final_stats() {
  for (size_t i = 0; i < histograms.size(); ++i) {
    std::string name = fmt::format("histogram.{}.{}.csv", basename, i);
    histograms[i].print(name);
  }
  for (size_t i = 0; i < set_histograms.size(); ++i) {
    std::string name = fmt::format("set_histogram.{}.{}.csv", basename, i);
    set_histograms[i].print(name);
  }
}

