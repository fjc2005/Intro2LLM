# L01: Tokenizer and Embedding

> **Course Positioning**: This is the **first experiment** of the Intro2LLM course, and also the "minimum viable entry point" of the entire LLM system. In this experiment, you will transform a piece of human-readable text into a tensor representation that the model can consume in a stable, reproducible manner, laying the groundwork for the subsequent Transformer architecture.


In this lesson, we will follow this logical sequence:

- First, build the **minimum viable pipeline**: text -> token IDs -> vectors -> positional information
- Then, abstract layer by layer: BaseTokenizer -> BPE -> Byte-Level BPE
- Finally, turn "makeshift" into "maintainable and reproducible": batch processing, saving/loading, testing, and debugging paths

---

## Experiment Purpose

This experiment mainly explains the principles and implementation of the Tokenizer and the Embedding layer. Our LLM system needs to convert human-readable text into a numerical representation that computers can process, and then obtain a vector representation through the embedding layer. To allow the model to understand the sequential information of the text, a positional encoding mechanism must also be introduced.

### What You Will Learn in This Chapter

- **Tokenization Algorithms**: Understand the training logic, encoding/decoding logic, and the role of merge rules in BPE and Byte-Level BPE.

- **Engineering Abstraction**: Why a unified `BaseTokenizer` interface is needed; how batch/padding/mask collaborate.
- **Embedding Layer**: How discrete IDs become a learnable lookup table; how gradients backpropagate to weights.
- **Positional Encoding**: The extrapolation and additive injection of Sinusoidal; the rotary injection and relative position characteristics of RoPE.
- **Debugging Skills**: How to manually calculate minimum examples for verification, and how to use tests/breakpoints to locate shape and broadcasting errors.

### Project Progress You Will Advance in This Chapter

- **Complete** the full implementation of `tokenizer/base_tokenizer.py`, `tokenizer/bpe_tokenizer.py`, and `tokenizer/byte_level_tokenizer.py`.
- **Complete** the from-scratch implementation of `TokenEmbedding` in `model/embedding.py` (understand the essence of table lookup).
- **Complete** the implementation of `PositionalEncoding` (sine/cosine) and `RoPE` (Rotary Positional Encoding) in `model/embedding.py`.

---

## Experiment Requirements and Constraints

- This course provides a code framework with "**only interfaces, no implementations**": the `pass` statements you see are what you need to fill in.
- You must complete the implementation on top of the existing function/class declarations: **modifying interface signatures is not allowed** (parameter lists, return value types, class names/method names).
- The experiment guide does not provide any implementation code that can be directly copy-pasted; only natural language pseudocode and step descriptions are allowed.
- Your implementation must be consistent with the documentation comments in the repository; and it must be able to support usage in subsequent lessons.
- Prioritize correctness and readability first, then consider performance optimization (optimizations must not change the behavior).

---

## Theoretical Prerequisites: What Exactly Are You "Representing" in This Lesson

Before diving into implementation details, let's clarify the four core objects of lesson 1, and their "information types":

1. **Token**: The smallest processing unit after text is split (could be a word, subword, byte-mapped character, etc.). It is still a "symbol" and has no continuous geometric meaning.
2. **Token ID (Integer)**: The token's index in the vocabulary, which is a discrete integer. Its numerical value has no semantic meaning (ID=5 is not larger or more important than ID=6).
3. **Embedding (Vector)**: Maps discrete tokens into a continuous vector space, where distances/angles between vectors begin to carry semantic and statistical correlations.
4. **Position**: Sequence order information. The Transformer's attention mechanism is sensitive to "sets" but insensitive to "order," so positional information must be explicitly injected.


You can view the entire pipeline as a "representation type upgrade":

Symbol (text/token) -> Discrete Index (ID) -> Continuous Representation (embedding) -> Continuous Representation with Sequence Structure (position-aware embedding)

All subsequent model structures (attention, feed-forward, residual, normalization) operate on this last type of object.

---

## Project Structure and Execution Flow

### lesson1 Related Directory Structure

You will primarily work in these files:

- `tokenizer/base_tokenizer.py`: Unified interface, batch encoding, saving/loading, special token management
- `tokenizer/bpe_tokenizer.py`: Character-level BPE (training merges, encoding/decoding according to merges)
- `tokenizer/byte_level_tokenizer.py`: Byte-level BPE (byte<->Unicode mapping + reusing BPE)
- `model/embedding.py`: TokenEmbedding, Sinusoidal Positional Encoding, RoPE

### Execution Flow

Ultimately, this experiment requires you to complete the following pipeline:

1. Text input `text`
2. `tokenizer.encode(text)` yields `token_ids` (integer list/tensor)
3. `TokenEmbedding(token_ids)` yields `token_vectors` (floating-point tensor)
4. Positional Encoding:
   - Option A: `PositionalEncoding(token_vectors)` direct additive injection
   - Option B: RoPE injected into attention's Q/K (implement the RoPE body in this experiment; subsequent attention modules will call it)

You should always keep the "input/output contracts" in mind:

- Tokenizer: String <-> ID sequence (savable, loadable, batch processing alignable)
- Embedding: ID tensor -> Vector tensor (gradients can backpropagate)
- PE/RoPE: Does not change the batch/seq dimensions, only injects positional information (numerically stable)

---

## List of Interfaces You Will Implement

### File 1: `tokenizer/base_tokenizer.py`

You need to complete:

- `BaseTokenizer.__init__`
- `BaseTokenizer.vocab_size`
- `BaseTokenizer.encode_batch`
- `BaseTokenizer.save`
- `BaseTokenizer.load`
- `BaseTokenizer.tokenize`
- `BaseTokenizer.convert_tokens_to_ids`
- `BaseTokenizer.convert_ids_to_tokens`

### File 2: `tokenizer/bpe_tokenizer.py`

You need to complete:

- `BPETokenizer.train`
- `BPETokenizer.encode`
- `BPETokenizer.decode`
- `BPETokenizer._pretokenize`
- `BPETokenizer._postprocess`
- `BPETokenizer.get_vocab`
- `BPETokenizer.tokenize`
- `BPETokenizer.save`
- `BPETokenizer.load`

### File 3: `tokenizer/byte_level_tokenizer.py`

You need to complete:

- `ByteLevelTokenizer.__init__`
- `ByteLevelTokenizer._create_bytes_to_unicode`
- `ByteLevelTokenizer._bytes_to_unicode`
- `ByteLevelTokenizer._unicode_to_bytes`
- `ByteLevelTokenizer.train`
- `ByteLevelTokenizer.encode`
- `ByteLevelTokenizer.decode`
- `ByteLevelTokenizer.save`
- `ByteLevelTokenizer.load`

### File 4: `model/embedding.py`

You need to complete:

- `TokenEmbedding.__init__`
- `TokenEmbedding.forward`
- `PositionalEncoding.__init__`
- `PositionalEncoding._create_pe`
- `PositionalEncoding.forward`
- `RoPE.__init__`
- `RoPE._compute_cos_sin`
- `RoPE.rotate_half`
- `RoPE.apply_rotary_pos_emb`
- `RoPE.forward`

---

## Part 1: Tokenizer

### 1.1 Why We Need a Tokenizer

The input to a neural network is a tensor, not a string. A tokenizer must solve at least two things:

- **Discretization**: Splitting text into tokens and mapping them to integer IDs.
- **Encapsulation Conventions**: Special tokens (pad/eos/unk/bos) must be consistent throughout the entire project, otherwise, training and inference will have "protocol mismatches."

From an effectiveness perspective, the tokenizer also determines:

- Sequence length (affects compute power and context capacity)
- Vocabulary size (affects parameter count and softmax computation)
- OOV handling capability (whether it can cover arbitrary inputs)

### 1.1.1 "Granularity of Splitting" and Three Typical Schemes

Splitting text into tokens is essentially choosing a "discretization coordinate system." Comparison of common granularities:

- **Word-level**: Tokens are complete words
  - Pros: Intuitively semantic, short sequences
  - Cons: Severe OOV (new words/spelling variations/multilingual), huge and sparse vocabulary
- **Character-level**: Tokens are characters
  - Pros: Almost no OOV (as long as the character set covers it)
  - Cons: Sequences become very long, long-range dependencies are harder to learn; can still bloat for multilingual character sets
- **Subword-level**: Tokens are high-frequency fragments (BPE/WordPiece/Unigram, etc.)
  - Pros: A compromise between "word-level semantics" and "character-level coverage," the mainstream for modern LLMs
  - Cons: More complex to train and implement; tokenization rules affect downstream performance and interpretability

Byte-Level BPE can be seen as a subword-level scheme that "replaces the character-level foundation with a byte-level foundation": trading readability and some length overhead for coverage.

### 1.1.2 An Engineering Perspective: Tokenizer is a "Protocol", Not a "Preprocessing Gadget"

The reason the Tokenizer needs to be system-wide consistent is that it directly defines the discrete distribution of training data and the model's input space.

- The semantics of vocabulary IDs must be globally consistent, otherwise the same ID having different meanings in different modules will cause unrecoverable training errors.
- Strategies for special tokens (pad/eos/unk/bos) must be consistent, otherwise:
  - The model might treat padding as valid content to attend to
  - Generation cannot stop correctly (eos mismatch)
  - Training loss is polluted by "meaningless tokens"

### 1.2 BaseTokenizer: Unified Tokenizer Interface in the Project

`BaseTokenizer` is the base class for the tokenizer that our entire project depends on. Any subsequent training/inference code relies solely on its interface and doesn't care whether you are using BPE or Byte-Level.

#### 1.2.1 `BaseTokenizer.__init__` Initialization Method

Goal: You must ensure the following facts hold:

- `vocab` is the mapping from token strings to IDs
- `inverse_vocab` is the mapping from IDs to token strings (used for decode)
- Strings and IDs for `pad/eos/unk` are always available (even if an empty vocab is passed)
- `bos` is optional, but if configured, it must also have a valid ID

Natural Language Pseudocode:

1. Read `special_tokens`; if empty, fill in the default configuration (pad/eos/unk, bos can be None).
2. Read `vocab`; if empty, create an empty mapping.
3. Sequentially check whether the string for each special token is in `vocab`:
   - If yes: record its ID
   - If no: assign it a new ID and write it to `vocab`
4. Build `inverse_vocab` based on the updated `vocab` (ensure one-to-one correspondence).
5. Write and cache the IDs for each special token (pad/eos/unk/bos).

Supplement: How to assign an ID to a "newly added token"

- You cannot assume the IDs of the passed `vocab` are strictly continuous integers `0..N-1` (although they are in most cases).
- Therefore, "the ID of a new token" must meet two requirements:
  - It does not conflict with any existing ID (otherwise `inverse_vocab` will overwrite, and decode will be unstable).
  - The result is reproducible when repeatedly initialized on the same `vocab` (do not rely on unstable traversal orders).
- A common and stable strategy is: first collect the set of IDs already occupied in the current `vocab`; then select an "unoccupied new ID" (e.g., the next integer greater than the current maximum ID). You should explain the rule you chose and your reasoning in your report.

You need to explain in your report: why "special tokens must be added to the vocabulary" is a protocol issue, rather than an implementation detail.

#### 1.2.2 `encode_batch` (padding, truncation, attention_mask)

Goal: Batch processing must align sequences of different lengths and generate masks.

Natural Language Pseudocode:

1. Call `encode` on each text to get a list of ID sequences.
2. Determine the target length:
   - If `padding=False`: target length is their respective lengths (no alignment)
   - If `padding=True` or `padding="longest"`: target length is the maximum length in this batch
   - If `padding="max_length"`: target length is `max_length` (must be provided)
3. If `truncation=True` is enabled:
   - If `max_length` does not exist: provide a clear error prompt or defined behavior
   - If the sequence is too long: truncate according to convention (usually keeping the front, preserving eos if necessary)
4. For each sequence:
   - Pad the end with `pad_token_id` until the target length is reached
   - Generate `attention_mask`: 1 for original valid positions, 0 for padding positions
5. If `return_tensors` is specified:
   - Convert `input_ids` and `attention_mask` into corresponding tensors/arrays.

You need to explain in your report: why the mask must be consistent with the padding, and how subsequent attention uses it to mask out padding.

#### 1.2.3 `save` / `load`

Goal: Loading after saving should restore the exact same encode behavior.

Natural Language Pseudocode:

- `save`:
  1. Create the directory if it does not exist
  2. Write out `vocab.json` (token -> id)
  3. Write out `special_tokens.json` (strings for pad/eos/unk/bos)
  4. Write out `tokenizer_config.json` (other meta-information you consider necessary, e.g., type name, version number, etc.)
- `load`:
  1. Read the above files and verify all fields are present
  2. Call the constructor to restore the instance
  3. Verify the consistency of `vocab` and `inverse_vocab`

Hint: Persistence is one of the core aspects of "engineering correctness." Without it, you cannot stably reproduce experiments or debug.

#### 1.2.4 `vocab_size` Accessing Tokenizer's Vocabulary Size

Goal: `vocab_size` must be consistent with the contents of `vocab`, rather than a cached number.

Natural Language Pseudocode:

1. Get the number of tokens in the current `vocab` mapping.
2. Return it as the vocabulary size.

Boundary Reminder:

- If you allow the IDs in `vocab` to be non-continuous, the definition of `vocab_size` should still be "the number of tokens", not "maximum ID + 1". Otherwise, an abnormal behavior will occur where "the vocabulary has only 100 tokens, but the max ID=10000".

#### 1.2.5 `tokenize`

Goal: Semantically, `BaseTokenizer.tokenize` means "splitting text into a list of token strings," but the base class usually doesn't know the specific splitting strategy.

Natural Language Pseudocode (two reasonable choices, pick one and explain in your report):

- Choice A (Stricter): Explicitly raise a "not implemented/needs subclass implementation" error in the base class to prevent silent bugs from misuse.
- Choice B (Looser): Provide a "minimum viable default strategy," for example, treating the input as a single overall token (or splitting by whitespace). This makes writing demos easier, but make it clear this is just a fallback and not equivalent to BPE/Byte-Level.

Engineering Reminder:

- The tests for lesson1 will verify base class logic like `encode_batch` by "writing a minimal subclass that overrides tokenize". Therefore, you need to ensure that after the subclass overrides `tokenize`, common functions like `convert_tokens_to_ids/convert_ids_to_tokens` can work stably.

#### 1.2.6 `convert_tokens_to_ids` (OOV fallbacks to unk)

Goal: Map a list of token strings to a list of IDs, applying consistent handling for OOV.

Natural Language Pseudocode:

1. Initialize an empty list of output IDs.
2. Iterate through input tokens:
   - If the token is in `vocab`: append the corresponding ID
   - Else: append `unk_token_id`
3. Return the ID list

Boundary Reminder:

- Do not automatically add bos/eos/pad here; these are the responsibilities of `encode/encode_batch`.
- Do not perform any "string cleaning" here (like strip or lower); cleaning should belong to the tokenize/preprocessing stage, otherwise it will cause "the same token to be rewritten differently in different paths."

#### 1.2.7 `convert_ids_to_tokens` (Out-of-bounds/unknown IDs fallback to unk)

Goal: Map a list of IDs to a list of token strings, ensuring the decode path never crashes.

Natural Language Pseudocode:

1. Initialize an empty list of output tokens.
2. Iterate through input token_ids:
   - If the ID is in `inverse_vocab`: append the corresponding token string
   - Else: append the string representation of `unk_token`
3. Return the token list

Boundary Reminder:

- This "unknown ID fallback" is an important robustness convention: even if abnormal IDs appear in upstream tensors, you can get interpretable output instead of a direct error crashing the debug process.

---

## Part 2: BPE (Training and Execution of Character-Level "Merge Rules")

### 2.1 The Intuition Behind BPE

Splitting text down to the character level means you will never have OOV, but sequences become too long and semantics too weak. BPE compromises between the two via "frequency-driven merging":

- High-frequency fragments (e.g., common roots, affixes, common words) will be merged into longer tokens.
- Low-frequency fragments can still degrade into shorter tokens (even single characters).

### 2.1.1 What is BPE: From Compression to Subword Vocabularies

BPE (Byte Pair Encoding) was originally an idea in the field of data compression: repeatedly merge the most common adjacent pairs of symbols in a sequence into new symbols, thereby using a shorter sequence of symbols to represent the original data.

Transferred to NLP:

- "Symbols" can be characters (char-BPE) or byte-mapped characters (byte-level).
- The "merge rules" are simply a list of merges.
- The order of merges acts as an executable tokenization program.

### 2.1.2 Why BPE Mitigates OOV

The root cause of OOV is "word-level discretization being too coarse": once the vocabulary hasn't seen the word, it cannot be encoded.

BPE's strategy is: any new word can be broken down into smaller subword/character sequences; meanwhile, high-frequency fragments are merged to shorten the sequence.

So it doesn't "eliminate OOV," but rather downgrades OOV from "the whole word" to "smaller units," allowing encoding to always degrade to basic symbols.

### 2.1.3 The Two Stages of BPE: Training Merges and Executing Merges

Conceptually, distinguish between two things:

- **Training (learn merges)**: Gather a list of merge rules (merges) from corpus statistics and determine the final vocabulary.
- **Encoding (apply merges)**: Execute merges on new text according to priority, yielding a sequence of tokens.

This is similar to the separation between a "compiler/linker" and a "runtime": training builds the rules, encode executes the rules.

### 2.2 `BPETokenizer.train` (Training the merges list)

The goal of the training phase is to obtain two things:

- The vocabulary `vocab`
- The merge rules `merges` (a sequentially ordered list of adjacent token pairs)

Natural Language Pseudocode:

1. Initialization:
   - Collect the set of base symbols from the corpus (usually the set of characters).
   - Add special tokens to the vocabulary (following the BaseTokenizer protocol).
2. Tally the "Word Frequency Table":
   - Use `_pretokenize` to split the text into basic units (a common practice is splitting by words/punctuation).
   - Count the occurrences of each basic unit.
   - Represent each basic unit as a "sequence of symbols" (e.g., a character sequence).
3. Iteratively merge until `vocab_size` is reached or no further merges are possible:
   - Count the frequencies of all adjacent symbol pairs (must be weighted by "basic unit frequency").
   - Select the symbol pair with the highest frequency as the merge for this round.
   - If highest frequency < `min_frequency`: stop.
   - Record the merge into `merges`.
   - Execute this merge across the symbol sequences of all basic units (replace adjacent A, B with AB).
   - Add the new symbol AB to the vocabulary.
4. End of training:
   - Build `merges_dict` (pair -> sequence index) for fast comparison of merge priorities.

You must answer in your report: why the order of merges affects the final encoding result.

### 2.2.1 Pros and Cons of BPE (Theoretical Level)

Pros:

- **Strong Coverage**: Can encode new words, spelling variations, compound words.
- **Controllable Vocabulary**: Controls the size of the model's input space via `vocab_size`.
- **Sequence Length Compromise**: Shorter than character-level, longer than word-level, usually an acceptable engineering tradeoff.

Cons and Risks:

- **Greedy/Locality**: Training is driven by local frequencies and may not equate to "optimal language units."
- **Interpretability and Readability**: Subword boundaries may not align with human intuition.
- **Language and Corpus Bias**: Merges heavily rely on the training corpus distribution; changing domains may degrade tokenization quality.
- **Implementation Complexity**: Requires maintaining merge priorities and handling boundaries like spaces/punctuation.

### 2.2.2 Common Variants: WordPiece and Unigram (For Information Only)

You are implementing BPE in this experiment, but you should know it's not the only solution:

- WordPiece: More common in early BERT series, training objectives and merge strategies differ slightly, often paired with probabilistic/likelihood perspectives.
- Unigram: Starts from a larger candidate vocabulary and uses a probabilistic model to prune it; tokenization often uses Viterbi to find the optimal decomposition.

In modern LLMs, you will see different combinations like BPE, SentencePiece-Unigram, Byte-Level BPE concurrently.

### 2.3 `BPETokenizer.encode` (Encoding Text into IDs)

Natural Language Pseudocode:

1. Use `_pretokenize` to break the text into a list of basic units.
2. Execute BPE on each basic unit:
   - Initialize to the finest-grained symbol sequence (e.g., character sequence).
   - Repeatedly execute merges until no merges can be applied:
     - Find all adjacent symbol pairs in the current symbol sequence.
     - Find the one that "appears in merges and has the highest priority."
     - Merge that pair by position in the sequence (Note: adjacency changes after merging, requires continued scanning/updating).
3. After getting the list of token strings, map them to IDs:
   - If token is in vocab: take corresponding ID
   - If not: use `unk_token_id`
4. If `add_special_tokens=True`:
   - If `bos` is defined: add bos to the beginning
   - Add eos to the end (according to project conventions)
5. If `truncation` is enabled and length exceeds `max_length`: truncate according to conventions.

Suggestion: Manually calculate encode with a tiny example (a few words, a few merges) first to ensure your merge logic aligns with the order.

### 2.4 `BPETokenizer.decode` (Restoring Text from IDs)

Natural Language Pseudocode:

1. Map each ID back to a token string (use unk if missing).
2. If `skip_special_tokens=True`: filter out special tokens like pad/eos/bos/unk (based on config).
3. Concatenate the token strings to get intermediate text.
4. Call `_postprocess` to clean up tokenization markers and normalize spaces.

Note: The goal of decode is usually "readability" and "approximate restoration," not necessarily ensuring absolute byte-level consistency (only Byte-Level pursues strong reversibility).

### 2.5 `BPETokenizer._pretokenize` (Splitting Text into "Training/Encoding Units")

Goal: Provide a stable sequence of "basic units" for `train` and `encode` (commonly words/punctuation/whitespace segments).

Natural Language Pseudocode (recommend implementing a "minimal but stable" version):

1. Perform the lightest normalization on input text (optional): e.g., folding consecutive whitespaces into a single space, or leaving them as-is; the key is to explain in your report how your choice affects decode readability.
2. Split the text into a list of units, requiring:
   - Unit splitting is deterministic for the same input.
   - Results do not produce empty string units (or they are filtered out), preventing "empty tokens" in training/encoding.
3. Return the unit list.

Engineering Suggestion:

- Basic tests for lesson 1 will use "strings without spaces" to test the core merge behavior of BPE to avoid whitespace complexity. Therefore, a safe choice for `_pretokenize` is: when the text contains no whitespace, treat the whole paragraph as a single unit; when it does contain whitespace, split by whitespace and discard empty strings.

### 2.6 `BPETokenizer._postprocess` (Restoring Concatenated Tokens to Readable Text)

Goal: Clean up the intermediate text concatenated during the decode phase and restore it to a more natural readable form according to your conventions in `_pretokenize`.

Natural Language Pseudocode:

1. Receive the intermediate text (which comes from direct concatenation of token strings).
2. Clean up based on your tokenization marker conventions (if you introduced "word-initial markers / subword markers", remove them here).
3. Perform whitespace normalization (optional): e.g., folding multiple spaces, stripping leading/trailing spaces.
4. Return the cleaned text.

Boundary Reminder:

- If you did not introduce any extra markers on the encode side (e.g., completely direct concatenation by characters/subwords), then `_postprocess` can be very simple, but should still ensure valid characters are not mistakenly deleted.

### 2.7 `BPETokenizer.get_vocab` (Exposing Vocabulary for Inspection and Saving)

Goal: Return the current vocabulary mapping for external inspection/debugging/saving.

Natural Language Pseudocode:

1. Return `vocab` (you can choose to return the original object or a copy).
2. If returning the original object, explain in your report: modifying the vocabulary externally may break tokenizer consistency; how do you avoid this misuse in the project.

### 2.8 `BPETokenizer.tokenize` (Merge Only, No ID Mapping)

Goal: Reuse the same BPE merge logic as `encode`, but output a list of token strings instead of IDs.

Natural Language Pseudocode:

1. Use `_pretokenize` to split the text into a list of basic units.
2. Execute the same merge process as `encode` on each unit to obtain that unit's sequence of token strings.
3. Concatenate the token sequences of all units in order into a total token list.
4. Return the token list (bos/eos are usually not added here; whether they are added should be consistently defined in your report).

### 2.9 `BPETokenizer.save` / `BPETokenizer.load` (The Single Source of Truth for Saved Merges)

Goal: Loading after saving must restore the exact same sequence of merges and the exact same vocabulary, ensuring reproducible encode behavior.

Natural Language Pseudocode:

- `save`:
  1. Create the directory if it does not exist.
  2. Write out `vocab.json` (token -> id).
  3. Write out `special_tokens.json`.
  4. Write out `merges.txt` (save each merge's left and right tokens sequentially line by line; requiring that the pair can be unambiguously restored upon reading).
  5. (Optional) Write out `tokenizer_config.json` to record the tokenizer type and key hyperparameters.
- `load`:
  1. Read the above files and verify existence and field completeness.
  2. Restore `vocab` and `merges`, and use them to construct the tokenizer instance.
  3. Rebuild `merges_dict` (pair -> sequence index) and verify its consistency with `merges`.
  4. (Recommendation) Perform a minimal self-check: take a few random token/id pairs and check forward/backward mapping to ensure `vocab` and `inverse_vocab` match.

---

## Part 3: Byte-Level BPE (Making Any Unicode Encodable)

### 3.1 Why We Need Byte-Level

Character-level BPE can still encounter coverage issues or preprocessing fragility in multi-language, rare symbol, and emoji scenarios. The core strategy of Byte-Level is:

- First convert text into a UTF-8 byte sequence (0..255).
- Perform BPE in the byte space (base vocabulary is fixed at 256).
- Represent the byte sequence as a processable string via a bijection of "byte to printable Unicode."

The result of doing this is: theoretically no OOV will ever occur, because any string can ultimately be represented as bytes.

### 3.1.1 UTF-8 and the Meaning of "Byte-Level"

UTF-8 is a variable-length encoding:

- Common ASCII characters are usually 1 byte.
- Many non-Latin characters can be 2-4 bytes.
- Emojis are usually 4 bytes.

The core idea of Byte-Level is: do not perform tokenization directly on the "surface form of characters," but rather perform merging and modeling on the underlying, stable UTF-8 byte sequence.

This brings two direct impacts:

- Coverage: Any Unicode character can be encoded.
- Length: Some characters will expand into multiple bytes, so sequence length might increase (but BPE merging will partially offset this).

### 3.2 `_create_bytes_to_unicode` (Reversible Mapping)

Goal: Construct a mapping table that maps each byte value to a printable Unicode character, and the mapping must be reversible.

Natural Language Pseudocode:

1. Select a range of "naturally printable" byte values and map them directly to characters of the same code point (e.g., the common printable ASCII range).
2. For the remaining unprintable/conflicting bytes:
   - Sequentially allocate them to an uncommon, but stably representable Unicode range.
3. Finally obtain a one-to-one mapping (bijection) of 256 bytes.

Report Requirement: Explain why a bijection (reversibility) is a necessary condition for Byte-Level decode.

### 3.2.1 Why Have This "Byte -> Unicode" Mapping Step

You might ask: since we are down to bytes, why not just tokenize directly on bytes?

The reason is engineering interfaces and implementation convenience:

- Python string manipulation, regex, and persistence work more naturally at the Unicode string layer.
- Processing raw bytes directly is possible, but makes many text operations cumbersome.
- The GPT-2 style approach is to map 0..255 to 256 printable characters, allowing the "byte sequence" to safely pass through string processing pipelines.

The essence of this step is "representation layer transformation," not the tokenization algorithm itself.

### 3.3 `_bytes_to_unicode` / `_unicode_to_bytes`

Natural Language Pseudocode:

- `_bytes_to_unicode(text)`:
  1. Encode `text` into a byte sequence using UTF-8.
  2. Look up the mapping table for each byte value to get a Unicode character.
  3. Concatenate them into a new string and return (this is the "byte-level representation").
- `_unicode_to_bytes(text)`:
  1. Construct a reverse mapping (Unicode character -> byte value) based on the mapping table.
  2. Map the input string character-by-character back to a byte sequence.
  3. Decode the byte sequence back to original text using UTF-8 (decode errors handled by convention, e.g., replacement strategy).

### 3.4 `ByteLevelTokenizer.train/encode/decode` (Reusing BPE)

Core Idea: Byte-Level wraps a transformation layer before and after BPE.

Natural Language Pseudocode:

- `train`:
  1. Apply `_bytes_to_unicode` to each line of the training corpus.
  2. Call/reuse the BPE training process on the converted corpus (initial vocabulary should cover 256 base symbols).
- `encode`:
  1. Original text -> `_bytes_to_unicode` -> Perform BPE encode on the byte representation.
- `decode`:
  1. BPE decode yields byte representation string -> `_unicode_to_bytes` -> Original text.

Hint: `_pretokenize` in Byte-Level often no longer strongly depends on "word boundaries"; you need to ensure stable behavior in your implementation.

### 3.4.1 Pros and Cons of Byte-Level (Engineering Trade-offs)

Pros:

- **Virtually no OOV**: Complete coverage of the input space.
- **Cross-lingual Robustness**: Doesn't rely on space-based tokenization, more stable for mixed-language text.
- **Unified Implementation**: The same logic adapts to different character sets.

Cons:

- **Poor Readability**: Tokens might be "weird characters," making debugging and manual inspection more painful.
- **Length Overhead**: Some characters expand into multiple bytes; unmerged sequences are longer.
- **Higher Sensitivity to Boundary Behaviors**: Handling spaces, newlines, and invisible characters requires very clear conventions.

### 3.5 `ByteLevelTokenizer.__init__` (Initialize Byte Mapping, Ensure Global Consistency)

Goal: Ensure any instance has the same "byte value (0..255) <-> Unicode single character" bijective mapping, and this mapping is consistently used in encode/decode.

Natural Language Pseudocode:

1. First complete the parent class initialization (so vocab/merges/special_tokens protocols are in place).
2. If the class variable `BYTES_TO_UNICODE` has not been created:
   - Call `_create_bytes_to_unicode` to create the mapping.
   - Cache it in the class variable to avoid repeated construction per instance (ensure consistency and performance).
3. Prepare the reverse mapping (Unicode character -> byte value) on the instance for use by `_unicode_to_bytes`.

Boundary Reminder:

- This mapping must be "globally fixed": the same byte value mapping to different characters in different instances will cause decode to be irreversible after saving/loading.

### 3.6 `ByteLevelTokenizer._create_bytes_to_unicode` (Construct a Reversible Mapping for 256 Bytes)

Goal: Construct a mapping of exactly size 256: each byte value maps to a "single character" Unicode character; all output characters are distinct, and a reverse mapping can be constructed for strict reversibility.

Natural Language Pseudocode (GPT-2 style idea, described down to implementation granularity):

1. First, select a set of "naturally displayable / less problematic" byte values, and map them directly to characters of the same code point:
   - Typical practice includes most printable ASCII (excluding control characters) and a safe extended Latin character range.
2. For the remaining byte values (those not included):
   - Iterate through the byte values in ascending order.
   - Sequentially allocate a "new Unicode code point" for each byte, starting from some base and incrementing.
   - Key requirement: these newly allocated characters must not conflict with characters from Step 1.
3. Finally, verify three things:
   - The mapping table size is 256.
   - Every value is a string of length 1.
   - The values remain 256 after deduplication (ensuring bijection).

### 3.7 `ByteLevelTokenizer._bytes_to_unicode` (UTF-8 Byte Sequence -> "Printable Character Sequence")

Goal: Convert arbitrary Unicode text into a "byte-level representation string," so subsequent BPE can operate solely at the string level while retaining all information without loss.

Natural Language Pseudocode:

1. Encode the input text into a byte sequence using UTF-8.
2. For each byte value:
   - Look up the `BYTES_TO_UNICODE` mapping to get a single character.
3. Concatenate these characters in the original order into a new string and return.

### 3.8 `ByteLevelTokenizer._unicode_to_bytes` ("Printable Character Sequence" -> UTF-8 Byte Sequence -> Original Text)

Goal: Strictly restore the byte-level representation string to the original text, ensuring decode(encode(text)) is reversible (at least provided no extra post-processing breaks it).

Natural Language Pseudocode:

1. Prepare the reverse mapping `UNICODE_TO_BYTES` (inverted from `BYTES_TO_UNICODE`).
2. Iterate through each character of the input string:
   - Look up the reverse mapping to get the corresponding byte value.
   - Sequentially collect them into a byte sequence.
3. Decode the byte sequence back to text using UTF-8:
   - If you chose a "replacement strategy" to handle abnormal byte sequences, state this in your report; but for this experiment, if the input comes from `_bytes_to_unicode`, undecodable situations should not occur normally.
4. Return the decoded text.

### 3.9 `ByteLevelTokenizer.train` (Training Merges on Byte Representation)

Goal: First map the corpus to a byte-level representation, then run BPE's merge learning logic on that representation; the initial base vocabulary should cover the 256 byte characters.

Natural Language Pseudocode:

1. Execute `_bytes_to_unicode` line by line on the training corpus to get the "byte-level corpus".
2. Construct/confirm the initial base vocabulary contains 256 base symbols (each corresponding to a byte-mapped character).
3. Execute the BPE `train` pipeline on the byte-level corpus to obtain merges and the expanded vocab.
4. After training ends, ensure structures like `merges_dict` and `inverse_vocab` are consistent with the final results.

### 3.10 `ByteLevelTokenizer.encode` / `decode` (Adding a Reversible Transform Layer Before and After)

Goal: Perform BPE merging and vocabulary mapping on the "byte-level representation string," but the interface exposed to the user remains the original text.

Natural Language Pseudocode:

- `encode`:
  1. Original text `text` -> `_bytes_to_unicode` yields `byte_text`.
  2. Execute BPE tokenization/merge logic on `byte_text` to get a sequence of token strings.
  3. Map the sequence of token strings to IDs (OOV falls back to unk).
  4. Optionally add bos/eos and perform truncation per project conventions.
- `decode`:
  1. ID sequence -> sequence of token strings (optionally skip special tokens).
  2. Directly concatenate token strings into `byte_text` (Note: this concatenation happens in the "byte-level representation space").
  3. `byte_text` -> `_unicode_to_bytes` to restore original text.
  4. Return text.

Alignment Reminder with Test Contracts:

- When merges is empty and vocab covers the 256 base symbols, encode/decode should remain reversible for arbitrary Unicode text (tests include Chinese and emojis).

### 3.11 `ByteLevelTokenizer.save` / `ByteLevelTokenizer.load`

Goal: After saving/loading, the behavior and reversibility of encode/decode remain unchanged.

Natural Language Pseudocode:

1. The core content to save is similar to BPE (vocab, merges, special_tokens, necessary config).
2. `BYTES_TO_UNICODE` does not need to be flushed to disk directly (because it can be deterministically rebuilt by `_create_bytes_to_unicode`), but you must ensure the rebuilding algorithm and version are consistent.
3. During load, after finishing parent class restoration, re-initialize the byte mapping and reverse mapping to ensure `_bytes_to_unicode/_unicode_to_bytes` is usable.

---

## Part 4: Embedding (The Runtime Environment from "ID" to "Vector")

### 4.1 TokenEmbedding: Learnable Table Lookup

The Tokenizer outputs discrete IDs, but Transformer computation occurs in a continuous vector space. The essence of Embedding is:

- A learnable parameter matrix of shape `[vocab_size, hidden_size]`.
- Each token ID selects a row from it as that token's vector representation.
- Gradients backpropagate through the loss to update the accessed rows.


### 4.1.1 Why Embedding is Not One-Hot

The most intuitive discrete representation is one-hot: given vocabulary size `V`, a token is a vector of length `V`, with only one position being 1 and the rest 0.

But one-hot has two issues:

- Huge and sparse dimensionality, unsuitable as direct input to neural networks.
- Semantic geometry cannot be expressed: the one-hot distance between any two distinct tokens is almost identical, unable to reflect similarity.

What Embedding does can be understood as "learning a linear mapping from a one-hot space to a low-dimensional dense space." Table lookup is simply a highly efficient implementation of this linear mapping.

### 4.1.2 What Embedding Learns

Empirically, Embedding learns to assign similar vector directions and distances to tokens that appear in similar contexts (aligning with the distributional hypothesis).

In language models, Embedding serves as the entire model's "input coordinate system." If this coordinate system is learned poorly, all subsequent layers will have to expend extra capacity to compensate.

#### 4.1.1 `TokenEmbedding.__init__`

Natural Language Pseudocode:

1. Save `vocab_size`, `hidden_size`.
2. Create a learnable weight matrix (rows = vocabulary size, columns = hidden dimension).
3. Perform reasonable initialization (e.g., small-variance random initialization).
4. Register it as a model parameter so the optimizer can update it.

Engineering Reminder (related to subsequent weight tying):

- You must be able to access the "embedding weight matrix body" so you can do weight tying later (embedding shares weights with lm_head).

#### 4.1.2 `TokenEmbedding.forward`

Natural Language Pseudocode (explaining "table lookup" clearly):

1. Input `input_ids`, typically shaped `[batch, seq_len]`, and they should be integer IDs (not one-hot).
2. For each position in `input_ids`:
   - Treat the ID at that position as a command to "select the ID-th row of embedding_table".
   - Obtain a vector of length `hidden_size`.
3. Stack all position vectors back according to the original batch/seq structure to get the output tensor:
   - Shape is `[batch, seq_len, hidden_size]`.
4. Return this tensor.

Boundary Reminder:

- If `input_ids` contains out-of-bounds IDs, you can choose to let the framework throw an error (exposing data/tokenizer errors earlier) or fall back to unk (more robust). In this project, "out-of-bounds is an error" is more common, as the tokenizer should guarantee valid IDs; but you need to state your choice in your report.

Report Requirement: Explain why this step is a "table lookup," and why gradients can return to the weight matrix.

---

## Part 5: Positional Encoding (Injecting "Order" into Vectors)

### 5.1 Why Positional Encoding is Needed

The attention mechanism itself is permutation equivalent regarding token arrangement: if you just give it a bunch of vectors without positions, it cannot distinguish between "the 1st token" and "the 10th token." Positional encoding is giving a distinct signal to each position.

### 5.1.1 A Key Fact: Self-Attention Inherently Lacks Order

To understand intuitively: the core of attention is "computing correlations across a set of vectors and computing a weighted sum." If only given a set of vectors without any positional clues, what the model sees is more like a "set" rather than a "sequence".

Therefore, positional encoding is not "icing on the cake," but a necessary condition to turn the Transformer from an "unordered set processor" into a "sequence modeler."

### 5.1.2 Absolute Position vs Relative Position (You will encounter this repeatedly later)

- **Absolute Positional Encoding**: Each position `pos` has a positional vector `PE(pos)`, combined directly with the token representation (usually added).
  - Typical: Sinusoidal, Learned Positional Embedding
- **Relative Positional Encoding**: Attention weights or Q/K interactions explicitly depend on the relative displacement `pos_i - pos_j`.
  - Typical: RoPE, ALiBi, relative position bias (T5 style)

Modern LLMs prefer relative position-class methods (RoPE/ALiBi, etc.), usually because they extrapolate length and handle long contexts more stably.

### 5.2 Sinusoidal (Sine and Cosine Positional Encoding)

Characteristics:

- Fixed, non-learnable parameters.
- Extrapolatable to sequence lengths unseen during training (within reasonable limits).
- Injection method is usually "additive": input vector + positional vector.


Key points you need to implement:

- In the initialization phase, precompute the positional table of `[max_len, d_model]` and register it as a buffer (does not participate in gradients).
- During forward pass, truncate prefix by `seq_len` and broadcast-add it.

### 5.2.1 What is Sinusoidal: A "Frequency-Layered" Positional Table

Sinusoidal gives each position a unique phase through a combination of sin/cos at different frequencies. Intuitively:

- Low-frequency dimensions change slowly with position, providing coarse-grained position.
- High-frequency dimensions change rapidly with position, providing fine-grained position.

Combining these dimensions, position is encoded as a set of multi-scale "dial readings."

### 5.2.2 Why Design It This Way: Extrapolation and Computable Relative Displacement

Two common selling points of Sinusoidal:

- **No learnable parameters required**: Positional table is determined by formulas.
- **Extrapolatable**: Theoretically, you can compute positional encoding for any length (as long as numerically stable).

Additionally, under certain conditions, the model can linearly approximate relative displacement information (because sin/cos satisfy addition formulas), which is a key reason it was highly effective in early Transformers.

Natural Language Pseudocode (Creating the table):

1. Create position index `pos = 0..max_len-1`.
2. Create dimension indices (targeting even dimensions only).
3. Compute the frequency scaling term for each dimension (based on `10000` and dimension ratios).
4. Fill even dimensions with sin, odd dimensions with cos, stitching them into a complete table.

#### 5.2.3 `PositionalEncoding.__init__` (Precompute and register as buffer)

Goal: Compute the positional table once during initialization and guarantee it won't be updated by the optimizer.

Natural Language Pseudocode:

1. Call `_create_pe` to generate a positional table of shape `[max_len, d_model]`.
2. Register this positional table as a module buffer (i.e.: moves to device along with the model, saved to state_dict, but does not participate in gradients).
3. (Optional) Explain in your report: why use buffer instead of parameter (parameters get changed by training, destroying the "fixed formula" property).

#### 5.2.4 `PositionalEncoding._create_pe` (Generate positional table based on formula)

Goal: Strictly implement the sine/cosine formula, ensuring values and dimensions align correctly.

Natural Language Pseudocode:

1. Create a position index vector: contains `0..max_len-1`, organized as a "column vector" for broadcast multiplication with the frequency term.
2. Create frequency scaling term vector (corresponding to even dimensions only):
   - Frequency decays exponentially with dimension: low dims change slow, high dims change fast.
   - The length of this vector should be `d_model/2`.
3. Generate positional table matrix:
   - Initialize an all-zero matrix of shape `[max_len, d_model]`.
   - Fill even dimensions with `sin(position * frequency term)`.
   - Fill odd dimensions with `cos(position * frequency term)`.
4. Return this matrix.

Boundary Reminder:

- `d_model` is typically required to be even; if not even, you need to explicitly define your handling strategy (raise error or allow last dimension to miss a sin/cos pair) in implementation.

#### 5.2.5 `PositionalEncoding.forward` (Truncate by seq_len and broadcast-add)

Goal: Do not alter input shape, perform only additive injection.

Natural Language Pseudocode:

1. Read the sequence length `seq_len` of the input `x` (from the second dimension of `x`).
2. Extract the first `seq_len` rows from the positional table to get a segment of shape `[seq_len, d_model]`.
3. Broadcast this segment to the batch dimension, performing element-wise addition with `x`.
4. Return the added tensor.

Engineering Reminder:

- If input `x` dtype is not float32 (e.g., fp16/bf16), you must clarify the dtype strategy when adding positional table to input (e.g., cast positional table to `x`'s dtype before addition), otherwise implicit type promotion or precision differences might happen.

### 5.3 RoPE (Rotary Positional Encoding)

RoPE Intuition: Do not add positional signals to input. Instead, perform "position-based rotation" on Q/K in attention, making relative displacement manifest in the dot product result. Widely adopted by modern LLMs.


Key points you need to implement:

- `inv_freq`: Inverse frequency vector (dimension is `head_dim/2`).
- `cos/sin`: Computed based on positions and frequencies; ultimately must be broadcastable to Q/K shapes.
- `rotate_half`: Split the last dimension in half, construct the "rotated vector half".
- `apply_rotary_pos_emb`: Apply rotary formula to q/k.

### 5.3.1 What is RoPE: Treating Every Pair of Dimensions as a 2D Vector to Rotate

A common way to understand RoPE:

- Pair up the last dimension of `head_dim`, treating each pair as a 2D vector `(x1, x2)`.
- Different dimension pairs correspond to different rotation frequencies.
- At position `pos`, rotate each pair of 2D vectors by an angle (angle is proportional to `pos`).

So fundamentally, it introduces "position-dependent phases" in the representation space of Q/K.

### 5.3.2 Why Can RoPE Express Relative Position?

The key property of RoPE is: After applying position-dependent rotation to Q/K, their dot product correlation depends solely on the relative displacement `pos_i - pos_j` in some form (not their absolute position values).

Intuitively: You rotate the same vector twice separately, and the rotation difference determines how well they align; the difference corresponds to relative position.

This is why RoPE is often termed a "relative positional encoding".

### 5.3.3 Pros, Cons, and Engineering Parameters of RoPE

Pros:

- **Relative position friendly**: Fits the needs of autoregressive modeling better.
- **Great empirical performance on long context**: Adopted by many modern LLMs.
- **Few parameters**: The core is formula-generated cos/sin, needing no extra learnable position tables (varies slightly by implementation).

Cons and Cautions:

- **Implementation is error-prone**: Shape alignment and broadcasting errors are very common.
- **Hyperparameters like base affect extrapolation**: Different bases change rotation frequency distributions.
- **Low-precision stability**: NaN/Inf and numerical ranges must be carefully managed for long sequences.

#### 5.3.1 Shape Alignment Checklist (Draw before you write)

Common Attention Inputs:

- `q`: `[batch, num_heads, seq_len, head_dim]`
- `k`: `[batch, num_kv_heads, seq_len, head_dim]`

Target shapes for position-related tensors:

- `cos/sin` (base): `[batch, seq_len, head_dim]`
- `cos/sin` (broadcast to heads): Insert dimension of size 1 at heads dim to become `[batch, 1, seq_len, head_dim]`

Before implementing, you must clarify:

- Where your `head_dim` comes from (usually q's last dim).
- Is `position_ids` `[batch, seq_len]` or `[seq_len]`? How do you uniformly handle them?
- Which dimensions are allowed to broadcast, and which must be strictly equal?

#### 5.3.2 Numerical Stability Requirements

In long sequences (e.g., thousands of tokens) and low precision (fp16/bf16), RoPE easily encounters numerical issues. In implementation, ensure:

- No NaN/Inf produced.
- Rotation does not systematically change vector norm (theoretically orthogonal rotation, norm should be preserved; minor errors are allowed).

#### 5.3.4 `RoPE.__init__` (Construct inv_freq and constrain dim)

Goal: Prepare the inverse frequency vector `inv_freq` for each "2D dimension pair," used to map position IDs to rotation angles.

Natural Language Pseudocode:

1. Check if `dim` is even (because RoPE rotates "in groups of two dimensions"); if not, explicitly error out or offer consistent conventions.
2. Construct dimension pair indices `i = 0..(dim/2 - 1)`.
3. Compute inverse frequency for each dimension pair `i`:
   - Frequency varies exponentially as `i` increases; `base` controls the overall frequency scale.
4. Store the resulting `inv_freq` as a module buffer (not participating in gradients) so it migrates devices with the model.

#### 5.3.5 `RoPE._compute_cos_sin` (Position -> Angle -> cos/sin, ensuring broadcastable shapes)

Goal: Calculate `cos` and `sin` tensors that are broadcastable to Q/K based on `position_ids` and `inv_freq`.

Natural Language Pseudocode:

1. Unify `position_ids` shapes:
   - If input is 1D `[seq_len]`: Assume all batches share the same position sequence, prepend a batch dimension.
   - If input is 2D `[batch, seq_len]`: Use directly.
2. Compute angle matrix `angles`:
   - Perform "outer product-style combination" for each position `pos` and each dimension pair frequency `inv_freq`.
   - Result shape should be `[batch, seq_len, dim/2]`.
3. Expand angles to `dim`:
   - Since each dimension pair corresponds to two dimensions, you need to expand/copy the `[dim/2]` angles into an angle layout of `[dim]`, so the last dimension aligns with Q/K's `head_dim`.
4. Calculate `cos` and `sin`:
   - Calculate element-wise cosine and sine on the expanded angles.
   - Output shapes for both are `[batch, seq_len, dim]`.
5. Numerical stability strategy (recommended):
   - Under low-precision input (fp16/bf16), compute angles and sin/cos in higher precision (e.g., float32) first, then cast back to input dtype as needed.

#### 5.3.6 `RoPE.rotate_half` (Construct "90-degree rotated" auxiliary vectors)

Goal: Split the last dimension in half `(x1, x2)`, returning `(-x2, x1)`, used to implement the 2D rotation formula.

Natural Language Pseudocode:

1. Evenly divide input tensor `x` along the last dimension into a front half `x1` and a back half `x2` (each half size `dim/2`).
2. Construct new tensor:
   - Front half takes `-x2`
   - Back half takes `x1`
3. Concatenate back along the last dimension, return a tensor of the same shape as `x`.

#### 5.3.7 `RoPE.apply_rotary_pos_emb` (Broadcast cos/sin to heads dim and apply rotation)

Goal: Apply rotary positional encoding to `q` and `k`, maintaining identical input-output shapes.

Natural Language Pseudocode:

1. Clarify input shapes:
   - `q`: `[batch, num_heads, seq_len, head_dim]`
   - `k`: `[batch, num_kv_heads, seq_len, head_dim]`
   - `cos/sin`: `[batch, seq_len, head_dim]`
2. Make `cos/sin` broadcastable to heads dim:
   - Insert a dimension of size 1 at the second dimension of `cos/sin`, making it `[batch, 1, seq_len, head_dim]`.
3. Apply rotation formula (element-wise):
   - `q_rot` = element-wise product of `q` and `cos`, plus element-wise product of `rotate_half(q)` and `sin`.
   - `k_rot` follows similarly.
4. Return `(q_rot, k_rot)`, with shapes identical to inputs `q/k` respectively.

Verification Suggestion (great to include in the report):

- Check norm conservation: Pre- and post-rotation, the norm of the last dimension of `q` should be practically unchanged (allowing for small numerical errors).

#### 5.3.8 `RoPE.forward` (Stringing position_ids together for use)

Goal: Complete the full path of "calculate cos/sin -> apply to q/k" within the forward method.

Natural Language Pseudocode:

1. Read `seq_len`, `head_dim`, device/dtype info from `q`.
2. Call `_compute_cos_sin` to get `cos/sin`.
3. Call `apply_rotary_pos_emb` to get rotated `q/k`.
4. Return rotated `(q, k)`.

Reminder related to Test Contracts:

- If you shift `position_ids` by a constant overall offset (translating all positions together), RoPE's attention scores should remain unchanged (reflecting its relative position nature). This is a strong signal of correct implementation.

---

## Exercises (Requirements for the Experiment Report)

### Experiment Report Format Requirements

- Use Markdown, primarily text explanations.
- Answer all exercise questions.
- List the knowledge points you consider important in this experiment, and explain their relationship to LLM principle knowledge points.
- Record your implementation trade-offs (e.g., did you choose to error out or auto-correct for a certain boundary condition) and reasons.
- Include at least one complete debugging log (failure -> location -> fix -> verify).

### Exercise 1: Understanding the "Interface Contract" (Tokenizer/Embedding Protocol)

Please explain:

- Why the existence of `BaseTokenizer` acts as a "protocol layer" rather than a "code reuse trick."
- What the semantics of the four types of special tokens (pad/eos/unk/bos) are, respectively.
- What training/inference errors would occur if different modules have inconsistent conventions regarding eos/pad.

### Exercise 2: Manually Calculate BPE Merging Once (Deconstruct the mechanism of merges)

Use a tiny corpus (construct your own) to complete 2-3 merge hand-calculations, and answer:

- What is the highest frequency pair you counted, and why?
- What impact does stopping early with `min_frequency` have?
- How does changing the order of merges affect encode results (provide a counter-example)?

### Exercise 3: Reversibility of Byte-Level

Please explain:

- Why `_create_bytes_to_unicode` must construct a bijection.
- How do you verify that `_bytes_to_unicode` and `_unicode_to_bytes` are inverse processes?
- What is the main cost of Byte-Level compared to character-level BPE (answer from perspectives of sequence length, readability, preprocessing, etc.)?

### Exercise 4: The Essence of TokenEmbedding and Gradient Paths

Please explain:

- Why embedding is a "table lookup" rather than a normal linear layer.
- Why gradients return only to "certain rows" of the embedding weight matrix.
- Why weight tying (embedding sharing weights with lm_head) works, and what interface conventions it relies on.

### Exercise 5: Shapes and Broadcasting in RoPE

Please explain:

- What do you consider the target shapes of q/k/cos/sin to be, respectively?
- Which dimensions are matched via broadcasting, and which must be strictly equal?
- How do you verify that "rotation does not change the norm" (allowing for numerical error)?

### Exercise 6: Verifying Your Implementation

Please attach to your report:

- The verification command you ran (pytest or a minimal verification you built).
- A summary of key verification results.
- A bug you encountered: symptom, locating process, fix point, and post-fix evidence.

---

## Further Reading

- Sennrich et al., 2015: Neural Machine Translation of Rare Words with Subword Units (BPE)
- Vaswani et al., 2017: Attention Is All You Need (Sinusoidal)
- Su et al., 2021: RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)
- GPT-2 / RoBERTa's Byte-Level BPE practice materials (understanding byte mapping motivations)

---

## Frequently Asked Questions (FAQ)

### Q1: Why doesn't decode(encode(text)) necessarily strictly equal the original text?

Character-level BPE typically prioritizes "readability" and "stable tokenization," introducing whitespace normalization or tokenization markers; Byte-Level relies more heavily on strong reversibility.

### Q2: Why is BPE training so slow?

Merge iteration requires repeatedly counting adjacent pair frequencies. You can prioritize correctness first, then consider optimizations using more efficient data structures or caching (but do not alter the behavior).

### Q3: What is the most common bug in RoPE?

Almost always shape/broadcasting alignment errors. The solution is not to "blindly adjust," but to clearly write down the target shape first, then compare against the actual tensor dimension by dimension.