# Summary of the Method

The main idea differs from standard methods. Standard methods check if the EEG matches the whole sound wave.

> **This method says:** "The brain only cares about sudden changes (onsets). Let's only check if the brain noticed the specific moments when a new sound started."

## TARGET_FS = 64
Audio usually comes at **44,100 Hz** (very fast). Our brain's attentional waves (Delta/Theta bands) are slow (**1–8 Hz**). We don't need high-frequency data. We downsample everything to **64 Hz** to match the sampling rate of the processed EEG data later.

---

## Pre-processing the Sound Wave

### Gammatone filterbank filtering
Raw audio is a single squiggly line. But our ear (cochlea) breaks sound into frequencies. This function splits the audio into frequency bands (channels) using a **Gammatone Filterbank**.

### Non linear compressions
Human hearing is logarithmic, not linear. If you double the sound pressure, it doesn't sound "twice as loud" to you. Raising the signal to the power of **0.3** mimics how the hair cells in our ear compress loudness before sending it to the brain.

### Enveloping
You begin with this:
`Audio → passed through auditory filterbank → produces many frequency channels (e.g., 32 filters)`

So your data looks like this: `time × freqChannels`. It’s basically a mini spectrogram.

* **Step 1 (Resampling):** Reduces the sample rate to **64 Hz**. Because the speech amplitude envelope changes slowly and EEG is also in low frequencies. So 64 Hz is enough.
* **Step 2 (Averaging Across Frequency):** This takes the 32 frequency channels (or however many you had) and averages them into 1 channel. You collapse the whole spectrum → into one curve.
* **Step 3 (Squeezing):** Removes unused dimensions, not important conceptually.

The signal we have at the end is a **broadband speech envelope**, a single line at 64 Hz. This line represents: *"How loud / intense the speech is at each moment."*

It contains:
* onsets
* rises
* drops
* syllabic rhythm
* amplitude structure

---

## Event/Onset Detection

Finding the events/onsets is the most critical scientific part of the paper. We are not looking for loud moments; we are looking for **changes**. We are interested in when the sound starts, i.e., onsets.

### PeakRate Detection

* **Step 1 (Bandpass Filter 1-10 Hz):** 1-10 Hz is the syllabic rate of speech. Human speech generally has about 3 to 7 syllables per second. We filter out slow drifts (<1Hz) and fast jitter (>10Hz) to focus purely on the rhythm of the syllables.
* **Step 2 (The Derivative/Rate of Change):** We calculate the slope.
    * If the volume is constant (even if loud), the derivative is **0**.
    * If the volume suddenly spikes (e.g. a 'P' or 'K' sound), the derivative is **high**.
* **Step 3 (Half-Wave Rectification):** We only care when a sound starts (onset), not when it ends.
    * Rising slope (Sound starting) = **Positive Derivative → KEEP**.
    * Falling slope (Sound fading) = **Negative Derivative → DISCARD** (Set to 0).
* **Step 4 (The Threshold):** A "Peak" here is a local maximum in the rate of change.
    * **Real-world equivalent:** It is the exact millisecond a new syllable hits the ear.
    * **Adaptive Threshold:** By using standard deviation, the code adapts to the volume. It captures the "prominent" onsets and ignores tiny fluctuations.

---

## Glimpsed vs. Masked Classification

* **Glimpsed:** The attended speaker was louder than the background noise/distractor. The brain heard this.
* **Masked:** The distractor was louder than the attended speaker. The brain likely missed this.

### Method 1: The Naive Approach
* **How it works:** At the exact moment of the peak, which envelope is higher?
* **The Flaw:** Sound perception isn't instantaneous. The brain integrates sound over time. Just because the distractor was louder for 1ms doesn't mean the target was fully masked.

### Method 2: The Paper's Framework (Spectrograms)
* **Step 1 (Obtaining spectrograms):** Instead of looking at the broadband envelope, we look at the frequency content over time. This handles **Spectro-Temporal Masking**. The male voice might be masking the female voice in the low frequencies (100Hz) but not the high frequencies (2000Hz).
* **Step 2 (Taking a +/- 200 ms window):** The brain doesn't process sound in 0ms slices. It buffers sound. We look at the 200ms surrounding the peak event.
* **Step 3 (Glimpse ratio):**
    * **SNR Threshold:** We use **-4dB** (`glimpse_SNR`). This means the attended speaker doesn't even have to be louder than the distractor; they just have to be close enough to be audible.
    * **Calculation:** We count how many "pixels" in that 200ms spectrogram window are "visible" (i.e., the attended speaker is dominant).
    * **Final Decision:** If more than 50% of the spectrogram pixels are visible (`glimpse_ratio > 0.5`), we classify the event as **Glimpsed**. Otherwise, it is **Masked**.

---

## Easy analogy to compare these methods

### 1. The "Average" Trap (Method 1)
Imagine you are in a room with a loud **Air Conditioner** (Low humming noise) and a **Bird** (High pitched chirp).

* **The Physics:** The Air Conditioner is physically louder (more energy) than the bird.
* **The Naive Code (Method 1):** It looks at the total volume. It sees `Volume_AC > Volume_Bird`.
* **The Result:** It says the bird is **MASKED** (Hidden).
* **The Reality:** You can hear the bird perfectly fine! Why? Because the AC is low-frequency and the bird is high-frequency. They don't fight.

### 2. The "Frequency Bin" Solution (Method 2 - The Pro Way)
The "Pro" method uses a **Spectrogram**. Think of the Spectrogram as a grid or a checkerboard.

* Rows = Frequencies (Low to High).
* Columns = Time.

The code doesn't compare the total volume. It compares them **pixel by pixel**.

In a perfect frequency split (like Bird vs AC), **BOTH** can be classified as Glimpsed because they don't overlap. They each "own" enough of the spectrogram to pass the test.
