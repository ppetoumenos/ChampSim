#ifndef REPLACEMENT_IBRDP_H
#define REPLACEMENT_IBRDP_H

#include <vector>

#include "cache.h"
#include "modules.h"

// MAX and SAFE values for the IbRDPredictor confidence counters
constexpr uint32_t MAX_CONFIDENCE = 3;
constexpr uint32_t SAFE_CONFIDENCE = 0;

// Max value and quantization granularity for the prediction
constexpr uint32_t MAX_VALUE_PREDICTION = 15;
constexpr uint32_t QUANTUM_PREDICTION = 8 * 1024;

// Max value and quantization granulariry for the timestamp
constexpr uint32_t MAX_VALUE_TIMESTAMP = 7;
constexpr uint32_t QUANTUM_TIMESTAMP = 16384;

// Sampling Period and max reuse distance that the sampler must be able to hold
constexpr uint32_t SAMPLER_PERIOD = 4096;
constexpr uint32_t SAMPLER_MAX_RD = (MAX_VALUE_PREDICTION + 1) * QUANTUM_PREDICTION;

// Number of bits that we keep for the PC and the address
// We enforce these numbers of bits by calling TransformAddress and
// TransformPC from the entry-points of our code so that the only values we
// use throughout our code are limited to these numbers of bits
constexpr champsim::data::bits BITS_PC{20};
constexpr champsim::data::bits BITS_ADDR{26};

// Sets and associativity of the IbRDPredictor storage
constexpr uint32_t IBRDP_SETS = 16;
constexpr uint32_t IBRDP_WAYS = 16;

// Selective Caching controls the use of cache bypassing
#define SELECTIVE_CACHING

//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///
//---            INSTRUCTION BASED REUSE DISTANCE PREDICTOR               ---///
//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///
struct Entry {
  bool valid{false};         // Valid: 1 bit
  long tag{0};           // Tag: 20 bits PC - 4 bits for set indexing
  uint32_t prediction{0};    // Prediction: 4 bits (limited by MAX_VALUE_PREDICTION)
  uint32_t confidence{0};    // Confidence: 2 bits (limited by MAX_CONFIDENCE)
  uint32_t StackPosition{0}; // StackPosition: 4 bits (log2(IBRDP_WAYS))
}; // Total = 27 bits

class IBRDPredictor
{
private:
  std::vector<std::vector<Entry>> predictor; // Predictor Storage
  const uint32_t num_ways;                   // Associativity
  const uint32_t set_mask;                   // mask for keeping the set indexing bits of pc
                                             //    always == numsets - 1;
  const uint32_t set_shift;                  // # of bits that we shift pc, to get tag
                                             //    always == log2(numsets)

  // Find an entry associated with the given PC
  Entry* FindEntry(long pc);

  // Allocate a new entry
  Entry* GetEntry(long pc);

public:
  IBRDPredictor(uint32_t _num_sets, uint32_t _num_ways);

  // Retrieve the reuse distance prediction for the given pc, 0 if none
  uint32_t Lookup(long pc);

  // Update the predictor entry for this pc with this observation
  void Update(long pc, uint32_t observation);
};

//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///
//---                     REUSE DISTANCE SAMPLER                          ---///
//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///

class RDSampler
{
private:
  struct SamplerEntry {
    bool valid{false};        // Valid: 1 bit
    long pc{0};               // PC of the sampled access: 20 bits
    long address{0};          // Address of the sampled access: 26 bits
    uint32_t FifoPosition{0}; // Position in the FIFO Queue: log2(sampler_size) bits
                              //     = 5 bits
  };                          // Total: 52 bits

  const uint32_t size;          // Sampler size == SAMPLER_MAX_RD / SAMPLER_PERIOD
  const uint32_t period;        // Sampling Period == SAMPLER_PERIOD
  uint32_t sampling_counter{0}; // Counts from period-1 to zero.
                                // We take a new sample when it reaches zero
  std::vector<SamplerEntry> sampler;   // Sampler Storage
  IBRDPredictor& predictor;     // Reference to the IbRDPredictor

public:
  RDSampler(uint32_t _period, uint32_t max_rd, IBRDPredictor& _predictor);

  // Update the sampler (and potentially the predictor)
  void Update(long address, long pc, access_type type);
};

struct ReplState {
  uint32_t timestamp{0};
  uint32_t prediction{0};
};




// 1) We use the 17 bit accessesCounter instead of the 'timer' variable
//    because we wish to count the accesses caused only by loads and stores
// 2) We break the accessesCounter into a lower and a higher part, just
//    to make our lives easier: Since only the 3 higher order bits of
//    the accessesCounter are used directly by our policy, we keep them
//    separate from the lower 14 bits. One could very well merge the two
//    parts in one variable and just write some extra code to isolate
//    the three higher order bits.
class ibrdp : public champsim::modules::replacement
{
  long NUM_WAY;

  IBRDPredictor predictor;
  RDSampler sampler;
  uint32_t accessesCounterLow{0};  // Lower 14 bits of acccessesCounter
  uint32_t accessesCounterHigh{1}; // Higher 3 bits of accessesCounter
  const uint32_t set_shift;        // constant == log2(numsets)
  std::vector<std::vector<ReplState>> repl;

public:
  explicit ibrdp(CACHE* cache);
  ibrdp(CACHE* cache, long sets, long ways);

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip, champsim::address full_addr, access_type type);

  // void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr, access_type type);

  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr, access_type type, uint8_t hit);
  // void replacement_final_stats()
};

#endif
