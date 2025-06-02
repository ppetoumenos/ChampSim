#ifndef SAMPLER_RD_H
#define SAMPLER_RD_H

#include <map>
#include <random>
#include <unordered_map>
#include <vector>

#include "cache.h"
#include "modules.h"

#define SPARSE_HISTOGRAMS

// These parameters were chosen to replicate (somewhat) the StatStack configuration
constexpr uint64_t sampling_length{1'000'000};
constexpr uint64_t hibernation_length{14'000'000};
constexpr uint64_t samples_per_window{1'500};
constexpr double sampling_rate{static_cast<double>(samples_per_window) / static_cast<double>(sampling_length)};
constexpr uint64_t max_distance{(1 << 28) - 1};

constexpr uint64_t cacheline_sz{64};
constexpr uint64_t smallest_targeted_cache_kb{1};
constexpr uint64_t scaling{(smallest_targeted_cache_kb * 1024) / cacheline_sz};
constexpr bool linear_buckets{true};

#ifdef SPARSE_HISTOGRAMS
using HistStorage = std::map<uint64_t, uint64_t>;
#else
using HistStorage = std::vector<uint64_t>;
#endif

class HistVector
{
  public:
    HistVector() = default;
    void increment(uint64_t idx);
    auto get() const;
	void resize(size_t new_size, uint64_t value);

  private:
    HistStorage counts;
};


/**
 * Holds Reuse Distance histograms for a certain window of measurements
 * We hold separate histograms for each issuing PC, but also one aggregate one
 * Reuse distances are grouped in buckets to reduce the storage overhead
 * Buckets are calculated by outright ignoring a number of lower order bits (scale)
 * and then squashing the remaining bits using a partially-linear/partially-log2 transformation (granularity)
 */
class HistogramRD
{
  public:
    HistogramRD() = delete;
    /**
     * Only constructor for HistogramRD
     * Initialises all internal variables and resizes the aggregate histogram
     * @param _granularity basically how many buckets it takes to double the corresponding distance. 
     *        _granularity of 1 implies a log2 relationship between distance and bucket. 
     *        _granularity above a threshold implies an identity relationship.
     *        _granularity of zero ignores the granularity
     * @param _scale how much we scale down reuse distances before we process them
     * @param _max_distance the highest distance we can accurately store: larger ones are capped to _max_distance
     */
    HistogramRD(uint64_t _granularity, uint64_t _scale, uint64_t _max_distance);

    /**
     * Record a new reuse distance in our histograms
     * It will update both the aggregate and the per-PC histograms.
     * It might allocate a new histogram for the PC, if none exists
     * @param ip the address of the instruction that caused the previous access
     * @param distance the reuse distance
     */
    void increment(champsim::address ip, size_t distance);

    /**
     * Calculate the average reuse distance in the aggregate histogram
     * @return the average
     */
    uint64_t average() const;

    void print(const std::string& filename);

  private:
    /**
     * The granularity at which we capture reuse distances
     */
    const uint64_t granularity; 

    /**
     * The log2 of granularity. Useful for dividing by shifting
     */
    const uint64_t log_gran;

    /**
     * Quantity by which we scale down reuse distances
     */
    const uint64_t scale;

    /**
     * The maximum distance we can capture
     */
    const uint64_t max_distance;

    /**
     * The maximum bucket we can capture
     */
    uint64_t max_bucket;

    /**
     * The actual aggregate histogram storage
     */
    HistVector hist;

    /**
     * Per PC histogram storage
     */
    std::unordered_map<uint64_t, HistVector> ip_hist;

    /**
     * Quantize the reuse distance
     * It first scales the distance down,
     * then it quantizes it using a partially-log transformation
     * Scales down only, if granularity is zero
     * @param distance the distance to quantize
     * @return the bucket corresponding to the distance
     */
    uint64_t _quantize(uint64_t distance) const;

    /**
     * Quantize the reuse distance
     * The function we actually use most of the time to apply quantization,
     * Directly returns max_bucket, if the distance is above max_distance
     * Otherwise it delegates to _quantize
     * @param distance the distance to quantize
     * @return the bucket corresponding to the distance
     */
    constexpr uint64_t quantize(uint64_t distance) const;

    constexpr uint64_t unquantize(uint64_t bucket) const;
};


/**
 * A class sampling reuse distances from the access stream
 * The reuse distances from each window are accumulated in a set of histograms
 */
class samplerRD : public champsim::modules::replacement {
  public:
    samplerRD() = delete;
	explicit samplerRD(CACHE* cache);
    samplerRD(CACHE* cache, long _sets, long _ways);

    void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr, access_type type, uint8_t hit);
    void replacement_final_stats();

  private:
    struct SamplerEntry {
      champsim::address ip;
      uint64_t timestamp;
      uint64_t set_timestamp;
      uint32_t window_id;
    };

    std::string basename;

    std::mt19937 rng{};
    uint32_t current_window{0};
    uint64_t sampling_start{0};
    uint64_t sampling_end{sampling_length};

    uint64_t access_count{0};
    std::vector<uint64_t> set_access_count;

    std::unordered_map<uint64_t, SamplerEntry> samples;

    uint64_t granularity_full;
    uint64_t granularity_set;
    uint64_t scaling_full;
    uint64_t scaling_set;
    uint64_t max_distance_full;
    uint64_t max_distance_set;
	bool ignore_writes;

    std::vector<HistogramRD> histograms;
    std::vector<HistogramRD> set_histograms;
};
#endif
