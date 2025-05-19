// Reimplementation of "Instruction-based Reuse Distance Prediction Replacement Policy" by Petoumenos, Keramidas, and Kaxiras
// 1st JILP Workshop on Computer Architecture Competitions: Cache Replacement Championship, June 2010, Saint Malo (in conjunction with ISCA'10)
// The original code targeted the championship similator (early ChampSim?)
// This one partially reimplements IbRDP for the latest ChampSim version using modern C++
// Functionality should be close or identical to the original one
// Parameters are tuned for the original range of traces and caches,
//    so they're likely to be suboptimal now

#include <algorithm>
#include <cassert>
#include <vector>

#include "ibrdp.h"

// -----------------------------------------------------------------------------
// ---------------                HELPER FUNCTIONS               ---------------
// -----------------------------------------------------------------------------

// returns bits 21:2 of pc
// We ignore bits 0 and 1, even though the x86 architecture instructions are
// not alligned on word boundaries, because we believe that the possibility of
// two different memory instrunction starting in the same memory word is too
// low to be worth a couple of extra bits.
long TransformPC(champsim::address pc) {
	return pc.slice(champsim::dynamic_extent{BITS_PC + champsim::data::bits{2}, 2}).to<long>();
}

// returns bits 31:6 of the real address
long TransformAddress(champsim::address address) {
  return address.slice(champsim::dynamic_extent{BITS_ADDR + champsim::data::bits{LOG2_BLOCK_SIZE}, champsim::data::bits{LOG2_BLOCK_SIZE}}).to<long>();
}

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

// FindEntry searches the predictor to find an entry associated with the
// given PC. Afterwards it updates the LRU StackPositions of the entries.
Entry* IBRDPredictor::FindEntry(long pc)
{
  long set = pc & set_mask;
  long tag = pc >> set_shift;

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
Entry* IBRDPredictor::GetEntry(long pc)
{
  Entry* lru = nullptr;
  long set = pc & set_mask;
  long tag = pc >> set_shift;

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

IBRDPredictor::IBRDPredictor(uint32_t _num_sets, uint32_t _num_ways) : predictor(_num_sets), num_ways{_num_ways}, set_mask{_num_sets - 1}, set_shift{champsim::msl::lg2(_num_sets)}
{
  for (uint32_t set = 0; set < _num_sets; ++set) {
    predictor[set].resize(num_ways);
    for (uint32_t way = 0; way < num_ways; ++way)
      predictor[set][way].StackPosition = way;
  }
}

// Lookup returns a reuse distance prediction for the given pc
// If it finds one, it returns the prediction stored in the entry.
// If not, it returns 0.
uint32_t IBRDPredictor::Lookup(long pc)
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
void IBRDPredictor::Update(long pc, uint32_t observation)
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

//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///
//---                     REUSE DISTANCE SAMPLER                          ---///
//---------------------------------------------------------------------------///
//---------------------------------------------------------------------------///

// _max_rd is always 1 larger than the longest reuse distance not truncated   //
// due to the limited width of the prediction, that is equal to:              //
// (MAX_VALUE_PREDICTION + 1) * QUANTUM_PREDICTION                            //
// Based on that the RDSampler allocates enough entries so that it holds      //
// each sample for a time equal to _max_rd cache accesses                     //
RDSampler::RDSampler(uint32_t _period, uint32_t max_rd, IBRDPredictor& _predictor) : size{max_rd / _period}, period{_period}, sampler(size), predictor{_predictor}
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
void RDSampler::Update(long address, long pc, access_type type)
{
  // ---> Match <---

  // Search the sampler for a previous sample of this address
  // Stop when we've checked all entries or when we've found a previous sample
  auto entry_it = std::find_if(sampler.begin(), sampler.end(), [&](SamplerEntry& entry) { return entry.valid && entry.address == address; });

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
    entry_it = std::find_if(sampler.begin(), sampler.end(), [&](SamplerEntry& entry) { return entry.FifoPosition == size - 1; });

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

ibrdp::ibrdp(CACHE* cache) : ibrdp(cache, cache->NUM_SET, cache->NUM_WAY) {}

ibrdp::ibrdp(CACHE* cache, long sets, long ways)
  : replacement(cache), NUM_WAY(ways), predictor(IBRDP_SETS, IBRDP_WAYS), sampler(SAMPLER_PERIOD, SAMPLER_MAX_RD, predictor), set_shift{champsim::msl::lg2(sets)}, repl(sets)
{
  for (auto& entry : repl)
    entry.resize(ways);
}

long ibrdp::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip, champsim::address full_addr, access_type type)
{
  // new_prediction is zero, unless Selective Caching is activated. That forces
  // all conditions which control cache bypassing to be always false
  uint32_t new_prediction = 0;

#if defined(SELECTIVE_CACHING)
  if (type != access_type::WRITE)
    new_prediction = predictor.Lookup(TransformPC(ip));
#endif

  uint32_t now = 0;         // Current time
  uint32_t time_left = 0;   // The line's predicted time left until reuse
  uint32_t time_idle = 0;   // The line's elapsed time since last use
  uint32_t victim_time = 0; // The idle or left time for the victim_way

  // If the predicted quantized reuse distance for the new line has the
  // maximum value, it almost certainly doesn't fit in the cache
  long victim_way = NUM_WAY;

  if (new_prediction < MAX_VALUE_PREDICTION) {
    // We search the set to find the line which will be used farthest in
    // the future / was used farthest in the past
    for (uint32_t way = 0; way < NUM_WAY; ++way) {
      // ---> Un-Quantize all the needed variables <---

      // 'timestamp' refers to a point in the past, so it should be less
      // than 'accessesCounterHigh'. If this is not the case, it means
      // that the accesses counter has overflowed since the last access,
      // so we have to add to accessesCounterHigh 'MAX_VALUE_TIMESTAMP+1'
      if (repl[set][way].timestamp > accessesCounterHigh)
        now = UnQuantizeTimestamp(accessesCounterHigh + MAX_VALUE_TIMESTAMP + 1);
      else
        now = UnQuantizeTimestamp(accessesCounterHigh);

      uint32_t timestamp = UnQuantizeTimestamp(repl[set][way].timestamp);
      uint32_t prediction = UnQuantizePrediction(repl[set][way].prediction);

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
    if ((UnQuantizePrediction(new_prediction) > victim_time) && (type != access_type::WRITE))
      victim_way = NUM_WAY;
  }

  // This can happen if time_idle and time_left are zero for all entries in the set.
  // It is unlikely but it does happen
  if ((type == access_type::WRITE) && (victim_way == NUM_WAY))
    victim_way = 0;

  return victim_way;
}

/* called on every cache hit and cache fill */
void ibrdp::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr, access_type type, uint8_t hit)
{
  uint32_t prediction = 0;
  long myPC = TransformPC(ip);
  long myAddress = TransformAddress(full_addr);

  if (type == access_type::LOAD) {
    accessesCounterLow++;
    if (accessesCounterLow == QUANTUM_TIMESTAMP) {
      accessesCounterLow = 0;
      accessesCounterHigh++;
      if (accessesCounterHigh > MAX_VALUE_TIMESTAMP)
        accessesCounterHigh = 0;
    }
    sampler.Update(myAddress, myPC, type);
  }

  if (way == NUM_WAY)
    return;

  if (hit && type == access_type::WRITE)
    return;

  // Get the prediction information for the accessed line
  if (type == access_type::LOAD)
    prediction = predictor.Lookup(myPC);

  // Fill the accessed line with the replacement policy information
  // For Loads and Stores we update both fields
  // For Ifetches we update with real info only the timestamp field
  //    the prediction field is set to zero (==no prediction)
  // For Writebacks, we give dummy values to both fields
  //    so that the line will be almost certainly replaced upon
  //    the next miss
  if (type != access_type::WRITE) {
    repl[set][way].timestamp = accessesCounterHigh;
    repl[set][way].prediction = prediction;
  } else {
    repl[set][way].timestamp = 0;
    repl[set][way].prediction = MAX_VALUE_PREDICTION;
  }
}
