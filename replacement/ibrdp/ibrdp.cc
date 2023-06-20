// Reimplementation of "Instruction-based Reuse Distance Prediction Replacement Policy" by Petoumenos, Keramidas, and Kaxiras
// 1st JILP Workshop on Computer Architecture Competitions: Cache Replacement Championship, June 2010, Saint Malo (in conjunction with ISCA'10)
// The original code targeted the championship similator (early ChampSim?)
// This one partially reimplements IbRDP for the latest ChampSim version using modern C++
// Functionality should be close or identical to the original one
// Parameters are tuned for the original range of traces and caches,
//    so they're likely to be suboptimal now

#include <algorithm>
#include <cassert>
#include <optional>
#include <vector>

#include "../../inc/cache.h"
#include "../../inc/ooo_cpu.h"
#include "../../inc/util.h"

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
constexpr uint32_t BITS_PC = 20;
constexpr uint32_t BITS_ADDR = 26;

// Sets and associativity of the IbRDPredictor storage
constexpr uint32_t IBRDP_SETS = 16;
constexpr uint32_t IBRDP_WAYS = 16;

// Selective Caching controls the use of cache bypassing
#define SELECTIVE_CACHING

// -----------------------------------------------------------------------------
// ---------------                HELPER FUNCTIONS               ---------------
// -----------------------------------------------------------------------------

// returns bits 21:2 of pc
// We ignore bits 0 and 1, even though the x86 architecture instructions are
// not alligned on word boundaries, because we believe that the possibility of
// two different memory instrunction starting in the same memory word is too
// low to worth a couple of extra bits.
static inline uint32_t TransformPC(uint64_t pc) { return static_cast<uint32_t>((pc >> 2) & bitmask(BITS_PC)); }

// returns bits 25:0 of address (bits 31:6 of the real address,
// the argument address has already been stripped of the byte offset bits)
static inline uint32_t TransformAddress(uint64_t address) { return static_cast<uint32_t>((address >> LOG2_BLOCK_SIZE) & bitmask(BITS_ADDR)); }

// The rest of the helper functions are self-explanatory
static inline uint32_t QuantizeTimestamp(uint32_t timestamp) { return (timestamp / QUANTUM_TIMESTAMP) & MAX_VALUE_TIMESTAMP; }

static inline uint32_t UnQuantizeTimestamp(uint32_t timestamp) { return timestamp * QUANTUM_TIMESTAMP; }

static inline uint32_t QuantizePrediction(uint32_t prediction)
{
  prediction /= QUANTUM_PREDICTION;

  if (prediction < MAX_VALUE_PREDICTION)
    return prediction;
  return MAX_VALUE_PREDICTION;
}

static inline uint32_t UnQuantizePrediction(uint32_t prediction) { return prediction * QUANTUM_PREDICTION; }

//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///
//---            INSTRUCTION BASED REUSE DISTANCE PREDICTOR               ---///
//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///
class IBRDPredictor
{
private:
  struct Entry {
    bool valid{false};         // Valid: 1 bit
    uint32_t tag{0};           // Tag: 20 bits PC - 4 bits for set indexing
    uint32_t prediction{0};    // Prediction: 4 bits (limited by MAX_VALUE_PREDICTION)
    uint32_t confidence{0};    // Confidence: 2 bits (limited by MAX_CONFIDENCE)
    uint32_t StackPosition{0}; // StackPosition: 4 bits (log2(IBRDP_WAYS))
  };                           // Total = 27 bits

  std::vector<std::vector<Entry>> predictor; // Predictor Storage
  const uint32_t num_ways;                   // Associativity
  const uint32_t set_mask;                   // mask for keeping the set indexing bits of pc
                                             //    always == numsets - 1;
  const uint32_t set_shift;                  // # of bits that we shift pc, to get tag
                                             //    always == log2(numsets)

  // FindEntry searches the predictor to find an entry associated with the
  // given PC. Afterwards it updates the LRU StackPositions of the entries.
  Entry* FindEntry(uint32_t pc)
  {
    uint32_t set = pc & set_mask;
    uint32_t tag = pc >> set_shift;

    // Search the set, to find a matching entry
    auto entry_it = std::find_if(predictor[set].begin(), predictor[set].end(), [&](Entry& entry) { return entry.valid && entry.tag == tag; });

    // If we found an entry, update the LRU Stack Positions
    if (entry_it != predictor[set].end()) {
      for (auto& entry : predictor[set])
        if (entry.StackPosition < entry_it->StackPosition)
          entry.StackPosition++;

      entry_it->StackPosition = 0;
      return &(*entry_it);
    }
    return nullptr;
  }

  // GetEntry is called when we want to allocate a new entry. It searches for
  // the LRU Element in the list, and re-initializes it
  Entry* GetEntry(uint32_t pc)
  {
    Entry* lru = nullptr;
    uint32_t set = pc & set_mask;
    uint32_t tag = pc >> set_shift;

    // Search the set to find the LRU entry
    // At the same time, update the LRU Stack Positions
    for (auto& entry : predictor[set]) {
      if (entry.StackPosition == num_ways - 1)
        lru = &entry;
      else
        entry.StackPosition++;
    }
    assert(lru != nullptr);

    // Initialize the new entry
    lru->valid = true;
    lru->tag = tag;
    lru->StackPosition = 0;
    return lru;
  }

public:
  IBRDPredictor(uint32_t num_sets, uint32_t num_ways) : predictor(num_sets), num_ways{num_ways}, set_mask{num_sets - 1}, set_shift{lg2(num_sets)}
  {
    for (uint32_t set = 0; set < num_sets; ++set) {
      predictor[set].resize(num_ways);
      for (uint32_t way = 0; way < num_ways; ++way)
        predictor[set][way].StackPosition = way;
    }
  }

  // Lookup returns a reuse distance prediction for the given pc
  // If it finds one, it returns the prediction stored in the entry.
  // If not, it returns 0.
  uint32_t Lookup(uint32_t pc)
  {
    Entry* entry = FindEntry(pc);
    if (entry != nullptr)
      if (entry->confidence >= SAFE_CONFIDENCE)
        return entry->prediction;

    return 0;
  }

  // Update finds the entry associated with the given pc, or allocates a new one
  // and then it updates its prediction: If the observation is equal to the
  // prediction, it increases the confidence in our prediction. If the
  // observation is different than the prediction, it decreases the confidence.
  // If the confidence is already zero, then we replace the prediction
  void Update(uint32_t pc, uint32_t observation)
  {
    Entry* entry = FindEntry(pc);

    // If no entry was found, get a new one, and initialize it
    if (entry == nullptr) {
      entry = GetEntry(pc);
      entry->prediction = observation;
      entry->confidence = 0;
    } else { // else update the entry
      if (entry->prediction == observation) {
        if (entry->confidence < MAX_CONFIDENCE)
          entry->confidence++;
      } else {
        if (entry->confidence == 0)
          entry->prediction = observation;
        else
          entry->confidence--;
      }
    }
  }
};

//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///
//---                     REUSE DISTANCE SAMPLER                          ---///
//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///

class RDSampler
{
private:
  struct Entry {
    bool valid{false};        // Valid: 1 bit
    uint32_t pc{0};           // PC of the sampled access: 20 bits
    uint32_t address{0};      // Address of the sampled access: 26 bits
    uint32_t FifoPosition{0}; // Position in the FIFO Queue: log2(sampler_size) bits
                              //     = 5 bits
  };                          // Total: 52 bits

  const uint32_t size;          // Sampler size == SAMPLER_MAX_RD / SAMPLER_PERIOD
  const uint32_t period;        // Sampling Period == SAMPLER_PERIOD
  uint32_t sampling_counter{0}; // Counts from period-1 to zero.
                                // We take a new sample when it reaches zero
  std::vector<Entry> sampler;   // Sampler Storage
  IBRDPredictor& predictor;     // Reference to the IbRDPredictor

public:
  // _max_rd is always 1 larger than the longest reuse distance not truncated   //
  // due to the limited width of the prediction, that is equal to:              //
  // (MAX_VALUE_PREDICTION + 1) * QUANTUM_PREDICTION                            //
  // Based on that the RDSampler allocates enough entries so that it holds      //
  // each sample for a time equal to _max_rd cache accesses                     //
  RDSampler(uint32_t period, uint32_t max_rd, IBRDPredictor& predictor) : size{max_rd / period}, period{period}, sampler(size), predictor{predictor}
  {
    for (uint32_t i = 0; i < size; ++i)
      sampler[i].FifoPosition = i;
  }

  // This function updates the Sampler. It searches for a previously taken
  // sample for the currently accessed address and if it finds one it updates
  // the predictor. Also it checks whether we should take a new sample.
  // When we take a sample, if the oldest (soon to be evicted) entry is still
  // valid, its reuse distance is longer than the MAX_VALUE_PREDICTION so we
  // update the predictor with this maximum value, even though we don't know
  // its exact non-quantized reuse distance.
  void Update(uint32_t address, uint32_t pc, uint32_t type)
  {
    // ---> Match <---

    // Search the sampler for a previous sample of this address
    // Stop when we've checked all entries or when we've found a previous sample
    auto entry_it = std::find_if(sampler.begin(), sampler.end(), [&](Entry& entry) { return entry.valid && entry.address == address; });

    // If we found a sample, invalidate the entry, determine the observed
    // reuse distance and update the predictor
    if (entry_it != sampler.end()) {
      entry_it->valid = false;

      uint32_t position = entry_it->FifoPosition;

      uint32_t observation = QuantizePrediction(position * period);
      predictor.Update(entry_it->pc, observation);
    }

    // ---> Sample <---

    // It's time for a new sample?
    if (sampling_counter == 0) {
      // Get the oldest entry
      entry_it = std::find_if(sampler.begin(), sampler.end(), [&](Entry& entry) { return entry.FifoPosition == size - 1; });

      // If the oldest entry is still valid, update the
      // predictor with the maximum prediction value
      if (entry_it->valid)
        predictor.Update(entry_it->pc, MAX_VALUE_PREDICTION);

      // Update the FIFO Queue
      for (auto& entry : sampler)
        entry.FifoPosition++;

      // Fill the new entry
      entry_it->valid = true;
      entry_it->FifoPosition = 0;
      entry_it->pc = pc;
      entry_it->address = address;
      sampling_counter = period;
    }
    sampling_counter--;
  }
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
//    separated by the lower 14 bits. One could very well merge the two
//    parts in one variable and just write some extra code to isolate
//    the three higher order bits.
struct IBRDP_Policy {
  IBRDPredictor predictor;
  RDSampler sampler;
  uint32_t accessesCounterLow{0};  // Lower 14 bits of acccessesCounter
  uint32_t accessesCounterHigh{1}; // Higher 3 bits of accessesCounter
  const uint32_t set_shift;        // constant == log2(numsets)
  std::vector<std::vector<ReplState>> repl;

  IBRDP_Policy(uint32_t num_set, uint32_t num_way)
      : predictor(IBRDP_SETS, IBRDP_WAYS), sampler(SAMPLER_PERIOD, SAMPLER_MAX_RD, predictor), set_shift{lg2(num_set)}, repl(num_set)
  {
    for (auto& entry : repl)
      entry.resize(num_way);
  }
};

std::unordered_map<CACHE*, IBRDP_Policy> ibrdp;

void CACHE::initialize_replacement() { ibrdp.try_emplace(this, NUM_SET, NUM_WAY); }

uint32_t CACHE::find_victim(uint32_t cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t pc, uint64_t full_addr, uint32_t type)
{
  auto& policy = ibrdp.at(this);

  // new_prediction is zero, unless Selective Caching is activated. That forces
  // all conditions which control cache bypassing to be always false
  uint32_t new_prediction = 0;

#if defined(SELECTIVE_CACHING)
  if (type != WRITEBACK)
    new_prediction = policy.predictor.Lookup(TransformPC(pc));
#endif

  uint32_t now = 0;         // Current time
  uint32_t time_left = 0;   // The line's predicted time left until reuse
  uint32_t time_idle = 0;   // The line's elapsed time since last use
  uint32_t victim_time = 0; // The idle or left time for the victim_way

  // If the predicted quantized reuse distance for the new line has the
  // maximum value, it almost certainly doesn't fit in the cache
  uint32_t victim_way = NUM_WAY;

  if (new_prediction < MAX_VALUE_PREDICTION) {
    // We search the set to find the line which will be used farthest in
    // the future / was used farthest in the past
    for (uint32_t way = 0; way < NUM_WAY; ++way) {
      // ---> Un-Quantize all the needed variables <---

      // 'timestamp' refers to a point in the past, so it should be less
      // than 'accessesCounterHigh'. If this is not the case, it means
      // that the accesses counter has overflowed since the last access,
      // so we have to add to accessesCounterHigh 'MAX_VALUE_TIMESTAMP+1'
      if (policy.repl[set][way].timestamp > policy.accessesCounterHigh)
        now = UnQuantizeTimestamp(policy.accessesCounterHigh + MAX_VALUE_TIMESTAMP + 1);
      else
        now = UnQuantizeTimestamp(policy.accessesCounterHigh);

      uint32_t timestamp = UnQuantizeTimestamp(policy.repl[set][way].timestamp);
      uint32_t prediction = UnQuantizePrediction(policy.repl[set][way].prediction);

      // ---> Look at the future <---

      // Calculate Time Left until next access
      if (timestamp + prediction > now)
        time_left = timestamp + prediction - now;
      else
        time_left = 0;

      // If the line is going to be used farther in the future than the
      // previously selected victim, then we replace the selected victim
      if (time_left > victim_time) {
        victim_time = time_left;
        victim_way = way;
      }

      // ---> Look at the past <---

      // Calculate time passed since last access
      time_idle = now - timestamp;

      // If the line was used farther in the past than the previously
      // selected victim, then we replace the selected victim
      if (time_idle > victim_time) {
        victim_time = time_idle;
        victim_way = way;
      }
    }

    // If the reuse-distance prediction for the new line is greater than
    // the victim_time, then the new line is less likely to fit in the
    // cache than the selected victim, so we choose to bypass the cache
    if ((UnQuantizePrediction(new_prediction) > victim_time) && (type != WRITEBACK))
      victim_way = NUM_WAY;
  }

  // This can happen if time_idle and time_left are zero for all entries in the set.
  // It is unlikely but it does happen
  if ((type == WRITEBACK) && (victim_way == NUM_WAY))
    victim_way = 0;

  return victim_way;
}

/* called on every cache hit and cache fill */
void CACHE::update_replacement_state(uint32_t cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t pc, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{
  uint32_t prediction = 0;
  uint32_t myPC = TransformPC(pc);
  uint32_t myAddress = TransformAddress(full_addr);

  auto& policy = ibrdp.at(this);
  if (type == LOAD) {
    policy.accessesCounterLow++;
    if (policy.accessesCounterLow == QUANTUM_TIMESTAMP) {
      policy.accessesCounterLow = 0;
      policy.accessesCounterHigh++;
      if (policy.accessesCounterHigh > MAX_VALUE_TIMESTAMP)
        policy.accessesCounterHigh = 0;
    }
    policy.sampler.Update(myAddress, myPC, type);
  }

  if (way == NUM_WAY)
    return;

  if (hit && type == WRITEBACK)
    return;

  // Get the prediction information for the accessed line
  if (type == LOAD)
    prediction = policy.predictor.Lookup(myPC);

  // Fill the accessed line with the replacement policy information
  // For Loads and Stores we update both fields
  // For Ifetches we update with real info only the timestamp field
  //    the prediction field is set to zero (==no prediction)
  // For Writebacks, we give dummy values to both fields
  //    so that the line will be almost certainly replaced upon
  //    the next miss
  if (type != WRITEBACK) {
    policy.repl[set][way].timestamp = policy.accessesCounterHigh;
    policy.repl[set][way].prediction = prediction;
  } else {
    policy.repl[set][way].timestamp = 0;
    policy.repl[set][way].prediction = MAX_VALUE_PREDICTION;
  }
}

/* called at the end of the simulation */
void CACHE::replacement_final_stats() {}
