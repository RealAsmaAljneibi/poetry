# Deep research review of Soft-CER v2 for Khaleeji dialect variation in Gulf Arabic poetry

## Executive summary

The provided Soft-CER v2 is a **weighted character-edit evaluation framework** designed to reduce “false errors” when comparing ASR hypotheses against reference verses in Gulf Arabic poetry by giving **partial credit** for recognized **dialect-valid substitutions** (e.g., ك/چ, ج/ي, ق/گ) and by **normalizing orthographic noise** (diacritics, punctuation, hamza seat variants, final elongation letters). This design direction is strongly aligned with how Arabic ASR evaluations have historically handled “non-standard orthography” via normalization and mapping files, because Arabic writing exhibits multiple valid renderings for the same utterance. citeturn17view0turn13search2turn20search1

The main methodological risk is that **hand-assigned substitution costs** (especially when applied globally at the character level) can drift away from (a) the true **empirical confusion structure** of your ASR system, (b) the true **context-conditioned linguistics** of Gulf sound changes (many are conditioned by surrounding segments or morphology), and (c) poetry-specific conventions such as **القافية/الروي** and **الإشباع** (including “ألف/ياء الإطلاق” and related وصل/خروج patterns). All three can lead to systematic **over-credit** (accepting genuine recognition errors as “dialect”) or **under-credit** (penalizing legitimate poetic or dialectal variants), with impacts that may differ by sub-dialect, recitation style, and transcription practice. citeturn17view0turn12search16turn12search1turn15view2

The strongest evidence-backed components to keep and expand are:

- **Normalization as a first-class scoring tier**, consistent with established Arabic evaluation practice (e.g., normalizing Alef/Yaa/Taa Marbuta). citeturn17view0turn20search1  
- Treating “dialectness” as **structural equivalence**, ideally via established dialect orthography/phonology layers (CODA/CODA* + CAPHI) rather than a growing ad-hoc pair list. citeturn15view3turn20search14turn20search34  
- Re-estimating costs from your own **confusion matrices** (or learning/regularizing them), using classic edit-distance learning approaches and/or phonological-feature priors rather than fixed “articulatory” guesses. citeturn18view0turn20search9

The report below (a) synthesizes the relevant Gulf dialect and poetry literature, (b) diagnoses where global character-level costs can misbehave, (c) proposes **statistically grounded alternatives** (confusion-matrix weighting, EM-learned edit weights, Bayesian priors), (d) adds explicit **poetry-safe rules** for rhyme/line endings, and (e) provides tables for recommended costs, datasets, evaluation metrics, and experiments.

## Named project references used for Soft-CER

To make the methodology defensible, the repo now treats Soft-CER as a
**research-informed diagnostic** grounded in the following named sources rather than an arbitrary
character-cost table:

- **Arabic orthography normalization**: Habash et al. (2018), *Unified Guidelines and Resources for Arabic Dialect Orthography*. This is the main justification for normalizing spelling variants before scoring.
- **Standard ASR metric practice**: Hugging Face Audio Course, [Evaluation metrics for ASR](https://huggingface.co/learn/audio-course/en/chapter5/evaluation). This is used for the strict WER/CER framing and the explanation of insertions, deletions, and substitutions.
- **Weighted edit distance**: Fontan et al. (2016), *Using Phonologically Weighted Levenshtein Distances for the Prediction of Microscopic Intelligibility*. This motivates partial substitution costs instead of all-or-nothing character penalties.
- **Khaleeji/Kuwaiti phonology guidance**: Al Abdan (2018), [*الظواهر الصوتية في اللهجة الكويتية*](https://dn710704.ca.archive.org/0/items/phonetics_131/%D8%A7%D9%84%D8%B8%D9%88%D8%A7%D9%87%D8%B1%20%D8%A7%D9%84%D8%B5%D9%88%D8%AA%D9%8A%D8%A9%20%D9%81%D9%8A%20%D8%A7%D9%84%D9%84%D9%87%D8%AC%D8%A9%20%D8%A7%D9%84%D9%83%D9%88%D9%8A%D8%AA%D9%8A%D8%A9%20%D9%80%20%D8%B1%D8%B3%D8%A7%D9%84%D8%A9%20%D9%85%D8%A7%D8%AC%D8%B3%D8%AA%D9%8A%D8%B1%20%D9%80%20%D8%B9%D8%A8%D8%AF%D8%A7%D9%84%D9%86%D8%A7%D8%B5%D8%B1%20%D8%A2%D9%84%20%D8%B9%D8%A8%D8%AF%D8%A7%D9%86%20%D9%80%20%D8%AC%D8%A7%D9%85%D8%B9%D8%A9%20%D8%A2%D9%84%20%D8%A7%D9%84%D8%A8%D9%8A%D8%AA%202018%D9%85.pdf). This supports candidate dialect alternations found in Gulf/Kuwaiti speech traditions.
- **Emirati affricate variability**: Szreder and Derrick (2024), *Phonological conditioning of affricate variability in Emirati Arabic*. This supports the claim that some Gulf alternations are real but context-conditioned, so Soft-CER should stay conservative.
- **Poetry-domain semantic layer**: Qarah, *AraPoemBERT: A Pretrained Language Model for Arabic Poetry Analysis*. This motivates keeping semantic similarity separate from strict CER rather than merging meaning into the edit score itself.

### Repo interpretation of those references

These sources justify a three-part stance:

1. **Strict CER/WER remain the headline accuracy metrics.**
2. **Soft-CER is only a post-hoc diagnostic layer for orthographic and dialect-sensitive variation.**
3. **Every low-cost substitution should be treated as conditional and revisable, not as a universal Gulf rule.**

## What the provided Soft-CER v2 is doing

### Metric family and its relationship to established ASR scoring

Character Error Rate (CER) and Word Error Rate (WER) are standard ASR metrics computed via Levenshtein alignment (insertions/deletions/substitutions). In Arabic ASR evaluations, it is common to apply **pre-scoring normalization** (drop diacritics/punctuation; normalize Alef/Yaa/Taa Marbuta) and sometimes apply a **global mapping file (GLM)** for common orthographic variants, because otherwise multiple “correct” spellings are unfairly counted as errors. citeturn17view0turn20search1

Soft-CER v2 extends this philosophy by replacing uniform substitution penalties with a **weighted substitution matrix** where select letter substitutions incur fractional costs (e.g., 0.15 instead of 1.0), intending to reflect dialectal phonological equivalence (such as /k/ affrication) and reduce penalties for predictable orthographic drift. This aligns conceptually with “phonologically weighted Levenshtein” work that assigns substitution costs based on distinctive-feature differences rather than treating every phoneme switch equally. citeturn20search9turn18view0

### Internal structure implied by the document

Based on the provided document’s descriptions, Soft-CER v2 operates as a **multi-tier evaluation**:

- A strict tier that measures raw character accuracy without normalization.
- A normalized tier where orthographic noise is removed (diacritics, punctuation, hamza/alef variants, digraph handling, line-end extensions).
- A dialect-aware tier (Soft-CER proper) that adds word-level equivalences and weighted Levenshtein substitutions.
- A semantic tier using AraPoemBERT cosine similarity (embedding-based verse similarity rather than surface-form similarity). citeturn19search0turn19search2turn19search5

This tiering is methodologically sound because it separates:  
(1) “exact transcription faithfulness,” (2) “orthographic standardization,” (3) “dialect tolerance,” and (4) “meaning preservation.” Similar multi-view evaluation is standard in Arabic evaluations where one reports multiple WER/CER variants (original text vs punctuation/diacritics stripped vs GLM vs normalized). citeturn17view0

### Practical limitation to name explicitly

Because only the methodology document was provided and **no paired ASR hypothesis/reference datasets** (nor audio) were included, this review cannot compute your empirical confusion matrices or validate frequency claims inside your own domain (Nabati/Gulf poetry recitation). Where the report recommends corpus-driven recalibration, it provides concrete procedures and experiments that require those files.

## Evidence base on Khaleeji dialect variation and Gulf poetry prosody that matters for Soft-CER

### Dialect variation in the entity["organization","Gulf Cooperation Council","regional political bloc"] area that directly affects grapheme-level scoring

Gulf Arabic is commonly described as the everyday spoken variety across the southern Gulf region, with substantial internal variation across entity["country","Kuwait","country"], entity["country","Saudi Arabia","country"], entity["country","Bahrain","country"], entity["country","Qatar","country"], entity["country","United Arab Emirates","country"], and entity["country","Oman","country"], and also by town/tribe and by urban/Bedouin social histories. citeturn9view0turn15view1

The Soft-CER-relevant point is that many “dialect equivalences” are:

- **Real and well-attested**, but  
- **not universal across all Gulf varieties**, and  
- often **conditioned** (phonological context, morphological position, or sociostylistic register). citeturn10view2turn8view0turn12search10turn14search8

These realities motivate (a) *conditional rules* and (b) *data-driven cost estimation* rather than a static, global substitution table.


### High-confidence dialect phenomena with direct mapping to the Soft-CER substitution list

The following phenomena have strong documentation in Gulf/Najdi/Emirati/Kuwaiti linguistics and are highly relevant to CER-like scoring:

**Affrication of /k/**  
Affrication processes for /k/ are widely discussed for Peninsula varieties. Work on Najdi/Qasimi Arabic shows /k/ affrication is favored in particular phonological environments (notably front vowels) and is also shaped by social variables and dialect leveling processes. citeturn12search10turn8view0turn8view1  
Separately, Emirati Arabic work documents optional realization patterns of /k/ as [tʃ] (and /dʒ/ as [j]) and explicitly investigates the phonological conditioning. citeturn8view2  
A Gulf Arabic reference grammar explicitly notes a “k → c” change in the second-person feminine singular suffixed pronoun, giving examples that effectively correspond to “ك/چ” behavior at the morphological level. citeturn10view2

**/dʒ/ realized as [j] (ج ↔ ي)**  
A Gulf Arabic reference grammar notes that “literary /j/ corresponds to GA /y/ in most instances,” providing examples such as “mosque” and “man,” which directly motivates very low-cost ج↔ي substitutions for Gulf outputs. citeturn10view3turn14search8turn8view2

**/q/ variation (ق realized as [g] and other outcomes)**  
A Gulf Arabic reference grammar includes explicit appendices noting forms where “literary q changes into g” and provides example lexemes; this supports treating ق↔(g-realization) as dialectal rather than “wrong” in many Gulf contexts. citeturn10view1turn10view0  
Modern dialect modeling work using CODA/CAPHI highlights that the same underlying consonantal root can surface differently across dialects (e.g., q-l-b realized as /qalb/, /ʔalb/, or /galb/ across different regions), which is precisely the type of equivalence Soft-CER tries to capture. citeturn20search14turn20search34

**Interdental merges (ث/ذ shifting to س/ز or to stops)**  
Across Arabic varieties, interdental reflexes are a classic source of dialect variation; dialect contact literature also indicates that such features can shift under social contact and register pressures. citeturn3search27turn8view4  
For scoring, the key is that ث~س and ذ~ز (and sometimes ث~ت) cannot be treated as “always equivalent” without controlling for target dialect, but they are common enough to justify low-to-moderate costs once empirical evidence supports them in the specific corpus.

### Gulf poetry prosody and “musical line endings” that must not be penalized

In Arabic prosody and rhyme theory, the end of the poetic line includes structured components: **الروي** (main rhyme consonant), as well as optional letters/sounds such as **الوصل** (including long vowels or certain pronouns) and **الخروج** (often created by “إشباع” of هاء الوصل). These mechanisms can produce **surface letter differences** at the end of the line that are prosodically legitimate. citeturn12search8turn12search16turn12search12turn12search15

For Nabati and Gulf vernacular poetry, scholarship notes that Nabati prosodic templates and rhyme behavior can differ from classical expectations (including rhyme structures across hemistichs and patterns like مزدوج القافية). citeturn15view2turn8view3  
Additionally, practical Arabic-language guidance on writing Nabati poetry explicitly discusses **إشباع حركة حرف الروي** and the distinction between the phonetic necessity of lengthening and its orthographic representation—exactly the kind of phenomenon that can lead to “extra ا/ي/و/ه” at line ends. citeturn12search1

A scoring scheme that ignores these rhyme mechanics will systematically over-penalize correct poetic outputs, especially for ASR transcripts where line-final vowels and pauses can change transcription choices.

## How to estimate and validate Soft-CER costs statistically

### Why fixed “phonemic” costs are fragile without calibration

A static table assigns the same penalty to a substitution regardless of:

- how often the ASR system actually confuses these characters,
- whether the substitution is dialectal in this corpus (speaker population, register, performance style),
- whether it is conditioned by phonological context (e.g., /k/ affrication environments),
- whether it is specifically a **poetic-line-end** artifact rather than a segmental difference.

Because none of these are constant across corpora, the principled way to set weights is to estimate them from aligned data and to validate them against human judgments and downstream metrics (e.g., retrieval relevance).

### Confusion-matrix–derived costs (recommended default)

Given a development set of (reference, ASR output) pairs, compute alignments and estimate a character confusion matrix:

- \( C_{a,b} = \#\{\text{times ref char }a\text{ aligns to output char }b\} \)
- \( P(b|a) = \frac{C_{a,b} + \alpha}{\sum_{b'} C_{a,b'} + \alpha|V|} \) (Dirichlet/Laplace smoothing)

Then define a substitution cost such as:

- **Negative log-probability**: \( \text{cost}(a\to b)=\lambda\cdot(-\log P(b|a)) \), scaled to [0,1] by dividing by a constant or using min–max normalization.
- Or **probability complement**: \( \text{cost}(a\to b)=1-P(b|a) \) (simple, interpretable).

This immediately anchors the cost table in your real system behavior, and it will naturally push rare substitutions (e.g., ض↔ذ, ص↔ز) toward higher cost unless they truly occur often in your domain.

This also aligns with established evaluation thinking: Arabic evaluations have long relied on normalization/mapping because “surface difference ≠ true error,” but the degree of forgiveness should depend on observed equivalence patterns. citeturn17view0turn20search1

### Learn edit-operation weights with EM (for principled edit-distance learning)

Classic work models edit distance as a stochastic transducer where substitutions/insertions/deletions are generated probabilistically and can be **learned from paired strings**, yielding a learned edit distance that outperforms untrained Levenshtein. citeturn18view0

In practice, you can treat your “dialect equivalences” as a **prior** and let EM (or modern differentiable variants) learn the actual operation probabilities from your corpus. The benefit over raw confusion matrices is that it learns in a way consistent with the global alignment model, not only local counts.

### Add a phonological-feature Bayesian prior (to prevent overfitting)

Confusion matrices alone can overfit to transcription quirks or small dev sets. A robust compromise is:

1. Define a **feature-based distance prior** (place, manner, voicing; emphatic vs non-emphatic; interdental vs alveolar).
2. Use that as a Bayesian prior over substitution probabilities (e.g., Dirichlet parameters larger for “near” substitutions).
3. Update with empirical counts.

This matches the rationale of phonologically weighted Levenshtein approaches that explicitly score substitutions by distinctive-feature differences. citeturn20search9

### Replace ad-hoc letter rules with CODA*/CAPHI where possible

CODA/CODA* aim to standardize dialect orthography with explicit meta-guidelines, and CAPHI provides a phonological representation layer that captures dialect pronunciations under a structured inventory. citeturn15view3turn20search34turn20search22

For Soft-CER, a powerful, less brittle direction is:

- Normalize both strings to a CODA-like spelling (or your own Gulf-poetry “CODA-lite”).
- Optionally convert to CAPHI-like phonological forms.
- Compute Levenshtein at the phonological level (or at least with a phonology-informed cost prior).

Recent dialect modeling explicitly motivates this: CODA and CAPHI expose structural equivalence across dialects, enabling normalization-aware methods. citeturn20search14turn20search0

## Recommendations: revised costs, rules to avoid penalizing legit dialect/poetry variants, and normalization strategy

### Guiding principles for “costs that are not arbitrary”

A defensible Soft-CER table should satisfy:

- **Empirical anchoring**: every low-cost rule should be supported by (a) linguistic evidence and (b) non-trivial empirical frequency in your actual ASR outputs.
- **Conditioning**: when a phenomenon is known to be context-conditioned (phonological or morphological), the “rule” should be conditional too.
- **Separation of concerns**: true orthographic normalization should be done as normalization (cost 0), not as “almost free substitutions,” to keep the cost matrix focused on genuine phonological equivalence.
- **Poetry-aware final-segment handling**: line-end additions tied to qāfiyah mechanics should be discounted in a controlled, explicit way, not left to accidental substitution paths. citeturn12search16turn12search12turn12search1

### Poetry-safe normalization rules that should be explicit

The following rules are recommended as **hard normalization (0 cost)** in the “normalized” and “dialect-aware” tiers unless you have a specific reason to preserve them:

- Strip diacritics and punctuation (standard evaluation practice). citeturn17view0turn20search1  
- Normalize Alef variants / hamza seats (أ/إ/آ/… → ا) and normalize final ى→ي (standard practice in Arabic normalization and evaluation). citeturn17view0turn20search1  
- Normalize ة/ه if your evaluation goal is “ASR semantic correctness” rather than morphological fidelity, noting that Arabic evaluation campaigns have explicitly reported normalized scoring that treats these differences as not-errors. citeturn17view0turn15view3  

**Additional poetry-specific line-end rules that are likely needed (not all are covered in the provided v2 list):**

- **Line-final إشباع letters**: drop trailing “ا/ي/و” when they function as “ألف/ياء/واو الإطلاق” (i.e., orthographic renderings of vowel lengthening at the rhyme). Arabic prosody sources explicitly define ألف الإطلاق as arising from إشباع حركة الروي (especially الفتحة), and discuss related وصل/خروج behavior. citeturn12search16turn12search8turn12search12turn12search1  
- **هاء الوصل / هاء السكت behavior**: optionally drop a final “ه” when it is a prosodic وصل rather than a lexical pronoun; prosody references describe هاء الوصل and how “الخروج” can be generated by its movement. citeturn12search8turn12search15turn12search27  
- **تنوين الترنم encoded as “ن”**: allow optional final “ن” where the poet orthographically realizes tarannum/nunation at line end (classical examples in prosody discussions treat such endings as a recognized poetic device). citeturn12search0turn12search16  

A practical implementation is: compute a “rhyme-normalized tail” for each line by removing a controlled set of optional tail characters after identifying the **last strong consonant** (candidate rawi). This keeps rhyme-sensitive content while removing ornamental orthography.

### Recommended cost table: original vs recommended

Because the strongest recommendation is **to learn and calibrate costs**, the table below should be interpreted as a **prior/default** (a starting configuration) designed to reduce obvious over-credit, introduce conditioning where linguistically required, and align the evaluation more closely with what is well-attested in Gulf dialect phonology and Arabic poetic practice.

| Pair / rule (v2) | Current cost | Recommended cost / rule | Why this change is more defensible |
|---|---:|---|---|
| أ ↔ إ ↔ آ ↔ ء ↔ ا ↔ ى | 0.05 | **Normalize to ا (0 cost)** | Major Arabic evaluation efforts explicitly normalize Alef-type differences as not-errors; treat this as normalization, not “soft substitution.” citeturn17view0turn20search1 |
| ى ↔ ي | 0.05 | **Normalize to ي (0 cost)** | Same rationale as Alef normalization; common in Arabic normalization pipelines. citeturn17view0turn20search1 |
| ه ↔ ة | 0.05 | **Tiered**: 0 in normalized tier; 0.05–0.15 only if you explicitly want morphological sensitivity | Arabic evaluation practice often reports a normalized score that removes this as an error; keeping it as a soft cost is only justified if you want to penalize morphological mistakes rather than transcription drift. citeturn17view0turn15view3 |
| و ↔ ؤ | 0.05 | **Normalize hamza-on-waw (0 cost)** | Hamza-seat spelling variants are classic normalization targets; treat as orthographic noise. citeturn20search1 |
| خ ↔ غ | 0.10 | **0.20 (default)**, or condition by voiced/voiceless context if you have audio | /x/ vs /ɣ/ is a real phonological contrast; giving near-free credit risks masking genuine ASR confusions. Without strong corpus evidence, keep this as “similar but not equivalent.” |
| غ ↔ ق | 0.20 | **Provisional**: raise to **0.35–0.50** unless your confusion matrix proves it’s common | Strongly corpus-dependent; if the real phenomenon is ق→[g] (ग) rather than ق→غ, this pair may over-credit. Use data-driven calibration; keep only if observed. citeturn10view1turn10view0 |
| ق ↔ ء | 0.25 | **Dialect-dependent**: keep 0.25 only if speakers/transcription allow q→ʔ; otherwise increase | Qaf can vary widely across Arabic; CAPHI shows /q/ vs /ʔ/ is a known cross-dialect equivalence but not necessarily “Khaleeji default.” Make it dialect-conditioned or data-calibrated. citeturn20search14turn20search34 |
| ق ↔ گ | 0.25 | **0.10–0.15** | Qaf→g is a core Peninsula/Gulf pattern and documented in Gulf Arabic descriptions; as dialect equivalence it deserves low penalty when evaluating dialect robustness. citeturn10view1turn10view0 |
| ق ↔ ي | 0.30 | **Remove global rule**; replace with **lexicon-based exceptions** if needed | A global ق↔ي rule can dramatically over-credit unrelated errors; handle any true cases as word-level variants (e.g., a small whitelist) rather than a character equivalence. |
| ت ↔ ط | 0.15 | **0.20** (default), or keep 0.15 if confusion matrix supports | Emphasis can be allophonic, but writing differences are semantically relevant; keep as “close” rather than “near-free.” |
| ض ↔ ظ | 0.15 | **Keep 0.15** (good prior) | Dialect discussions and descriptive work often treat emphatic contrasts as unstable/merged in some communities; a low penalty is defensible. citeturn8view4 |
| ض ↔ ذ | 0.20 | **Increase to 0.40–0.60** unless proven frequent | Much less robustly justified as a dialect equivalence; risk of over-credit is high. Use empirical counts. citeturn8view4 |
| ث ↔ س | 0.20 | **0.15–0.25**, and consider dialect conditioning | Interdental shifts are common across Arabic and can occur under contact; treat as partially equivalent but calibrate. citeturn3search27turn8view4 |
| ذ ↔ ز | 0.20 | **0.15–0.25**, and consider dialect conditioning | Same as above. citeturn3search27turn8view4 |
| ث ↔ ت | 0.20 | **0.20 (keep)** but consider conditioning | θ→t is attested in some varieties; keep moderate unless corpus says otherwise. citeturn8view4turn3search27 |
| س ↔ ش | 0.20 | **Conditioned** (0.20 only in specific lexical/phonological contexts); otherwise ≥0.40 | High risk of over-credit globally; treat as context-sensitive unless confusion matrix shows it is truly systematic in your poetry corpus. |
| ص ↔ س | 0.25 | **Conditioned by “emphasis spread / ibdal” context** (low cost only near emphatics/ق/غ/خ/ط), otherwise high | Classical/dialectal sources discuss sin→sad in specific environments (e.g., proximity to emphatics); encode the environment rather than giving blanket credit. citeturn6search2turn6search0 |
| ص ↔ ز | 0.25 | **Increase to 0.50–0.70** unless proven frequent | Likely to be over-forgiving unless your corpus explicitly demonstrates it. |
| ج ↔ ي | 0.15 | **0.10** | Very well-attested Gulf pattern; reference grammar explicitly notes broad j→y correspondences, and Emirati phonetics discusses [j] outcomes. citeturn10view3turn8view2 |
| ك ↔ چ / تش | 0.15 | **0.10–0.15** | Well-attested Gulf/Najdi affrication and documented in both descriptive and experimental work; low penalty is defensible. citeturn10view2turn8view2turn12search10 |
| ك ↔ تس | 0.20 | **0.15–0.25; ideally conditioned** | Najdi/Qasimi affrication patterns are conditioned (front-vowel environments, style/social effects); global equivalence is risky without constraints. citeturn12search10turn8view0 |
| لك ↔ لج | 0.25 | **Generalize to a pronoun/clitic rule** rather than a single token pair | If the intent is “2nd-person feminine enclitic realization,” handle it at the morphological/clitic level so it covers منك/منچ etc; this matches the evidence that k→affricate is strongly tied to the 2F.SG suffix in some descriptions. citeturn10view2 |
| ن ↔ م (final) | 0.30 | **Conditional**: allow only before labials / within known assimilation patterns; otherwise treat as normal | Assimilation n→m is a real process, but it is context-dependent; Arabic assimilation studies (including on Kuwaiti Arabic) treat such processes as conditioned, not global. citeturn24search23turn24search9 |
| All tashkeel | 0.00 | **Keep (hard strip)** | Standard normalization choice for many evaluation settings. citeturn20search1turn17view0 |
| All punctuation | 0.00 | **Keep (hard strip)** | Standard evaluation normalization. citeturn20search1turn17view0 |
| Final ا/ي extension | 0.00 | **Keep + extend to poetic وصل/خروج handling** (optional  ه/و/ن cases) | Prosody sources define line-end extensions from إشباع (including ألف الإطلاق and related mechanisms). Extend the rule set beyond ا/ي where your poems require it. citeturn12search16turn12search12turn12search1 |

### Core recommendation that reduces future table growth: move from “letter-pairs list” to “typed rules”

Instead of treating all substitutions equally as “pairs,” represent them as a small number of typed transformations:

- **Orthography normalization rules** (Alef/Yaa/Taa Marbuta; hamza seats; punctuation/diacritics; Unicode shaping).
- **Dialect phonology rules** (qaf realizations; k and j lenition/affrication; interdental reflexes).
- **Poetry prosody/rhyme tail rules** (إشباع, الوصل, الخروج, تنوين الترنم).
- **Context-conditioned assimilation rules** (n→m, emphasis spread).

This improves interpretability (you can report which rule fired how often), and it simplifies regression testing.

### Semantic tier: keep, but validate against poetry-specific semantic tasks

AraPoemBERT is a poetry-pretrained model explicitly designed for Arabic poetry tasks, trained on a large corpus of verses and evaluated across poetry-related tasks. Using it as a semantic similarity layer is a reasonable choice for “meaning-preserving” scoring beyond surface form. citeturn19search0turn19search2turn19search5

However, to keep the semantic tier “scientifically defensible” in an ASR evaluation context, validate that:

- Semantic similarity correlates with human judgments for your dialect/genre subset.
- The embedding similarity is not overly driven by shared rhyme tokens or boilerplate formulae common in Nabati.  
That calls for a small annotated set and correlation analyses (Spearman/Kendall), described below.

## Evaluation plan, datasets, metrics, and implementation steps

### Datasets and resources that can support Gulf-focused cost estimation and validation

The critical missing input for statistically grounded cost learning is a set of aligned (audio → ASR text) with gold references. If you do not yet have a poetry ASR dataset, the following resources can still help bootstrap language/dialect modeling, normalization conventions, and dialect coverage (though not all are poetry):

| Resource | Type | What it gives you | Why it matters for Soft-CER |
|---|---|---|---|
| Gumar corpus (≈110M words, 1,200 documents; sub-dialect annotated) | Text | Large-scale Gulf Arabic written variability + sub-dialect metadata | Useful for mining spelling variants and building lexicons/whitelists for word-level equivalence. citeturn16view0turn12search21 |
| Annotated Emirati subset of Gumar (≈200k words; spelling conventionalization + dialect ID) | Text + annotation | Concrete conventionalization guidelines and consistency targets | Helps design “CODA-lite for Gulf poetry” and evaluate normalization decisions. citeturn15view1 |
| Arabic MGB-3 (includes Gulf for dialect ID; hours per dialect for dev/eval) | Speech | Dialect-labeled audio, including Gulf category | Can be used to prototype cost learning from confusions before moving to poetry. citeturn7search3turn7search27 |
| ADI-20 (multi-dialect hours-scale dialect dataset) | Speech | Very large dialect identification corpus | Useful for dialect classifiers / selecting matched speakers, but may be restricted access. citeturn7search15 |
| Mixat (Emirati-English code-mixed speech) | Speech | Emirati speech audio and transcripts | Useful if your ASR sees code-switch; also helps tune Gulf acoustic confusions. citeturn7search9 |
| Kuwaiti dialect corpora (e.g., WhatsApp chats; structured story corpora) | Text | Kuwaiti lexical/orthographic variants | Useful for building word-level equivalence lists that are not safely captured by character pairs. citeturn7search18turn7search6 |

### Evaluation metrics you should report (and why)

To prevent metric gaming and to preserve interpretability, report at least these:

- **WER/CER (strict)**: the “what did the ASR literally output?” baseline. citeturn20search33  
- **CER normalized**: punctuation/diacritics + core letter normalization; aligns with Arabic evaluation best practice and gives a fairer baseline. citeturn17view0turn20search1  
- **Soft-CER**: your dialect/prosody-tolerant score (current v2 and revised/calibrated).  
- **Delta metrics**: (CER_normalized − SoftCER) to quantify how much “dialect tolerance” changes your evaluation; MGB-style evaluations similarly report multiple WER variants to show normalization effect sizes. citeturn17view0  
- **Poetry-tail error rate**: a specialized diagnostic: error rate restricted to last N characters of each hemistich/line, before and after “rhyme tail normalization.” This targets your stated goal (avoid penalizing musical endings). citeturn12search16turn12search12  
- **Semantic similarity (AraPoemBERT)**: retain as Tier 4, but report its correlation with human ratings on a held-out set. citeturn19search0turn19search2  

### Suggested experiments

The most informative experiments (and the ones that will justify costs scientifically) are:

**Ablation over scoring layers**  
Evaluate several conditions: Strict → Normalized → Soft-CER(v2) → Soft-CER(calibrated) → Soft-CER+PoetryTailRules. Measure effect sizes on both average score and ranking changes of systems.

**Data-driven cost learning vs. hand costs**  
Learn costs from confusion matrices and/or EM-learned edit distance, then compare:

- correlation with human judgments (pairwise or Likert similarity of hypothesis vs reference),
- stability across sub-dialects,
- sensitivity to line-end phenomena.

This is directly supported by the classic claim that learned edit distances can substantially outperform untrained Levenshtein in real tasks. citeturn18view0

**Dialect-conditioned scoring**  
Train a dialect classifier (even coarse Gulf vs non-Gulf, or country-level if available) and apply different substitution priors by dialect class. MGB-style dialect identification tasks and large dialect datasets provide the scaffolding for this approach. citeturn7search3turn7search15

**Rhyme/meter diagnostic set**  
Create a small hand-annotated set where judges mark: “same meaning,” “same meter,” “same rhyme,” to test whether your scoring aligns with poetic correctness rather than just lexical overlap. AraPoemBERT and other poetry NLP work show that meter/rhyme are measurable targets, but you still need domain-specific validation for Nabati. citeturn19search0turn19search8

### Implementation steps and workflow diagram

A practical workflow (designed so each step can be unit-tested and audited) is:

```mermaid
flowchart TD
A[Collect paired ref/hyp lines<br/>+ optional audio] --> B[Text normalization<br/>diacritics/punct/alef/yaa/...]
B --> C[Poetry tail normalization<br/>wasl/khروج/itlaq rules]
C --> D[Tokenization layer<br/>handle تش/چ and any multi-char graphemes]
D --> E[Alignment + confusion matrices<br/>char-level and/or CAPHI-level]
E --> F[Cost estimation<br/>P(b|a), EM learning, Bayesian smoothing]
F --> G[Soft-CER scoring]
G --> H[Validation<br/>human correlation + ablations + error analysis]
H --> I[Freeze versioned ruleset<br/>and publish diagnostics]
```

### Concrete code patterns to produce the requested visualizations (confusion matrix + spectrogram)

Because you did not provide audio or aligned hypothesis/reference files, the report cannot generate your actual spectrograms or confusion matrices. The following implementation patterns are what you would run once you supply those assets:

**Confusion matrix from alignments (character-level)**  
1) Align ref/hyp with dynamic programming.  
2) Accumulate aligned pairs into a matrix.  
3) Plot as an image.

**Spectrograms for critical minimal pairs**  
After you identify frequent confusions (e.g., ك↔چ, ق↔گ, ج↔ي), plot spectrograms of the corresponding audio segments to verify whether acoustic cues plausibly support the confusion (especially for emphatics and interdental variants).

(If you share a small sample set—~200 lines with audio + ref + hyp—I can convert these procedures into a reproducible notebook and produce the actual figures.)

### Final note on scientific defensibility

The main way to turn Soft-CER from a “reasonable heuristic” into a **publishable, defensible methodology** is to report:

1. **Ablations** showing what each normalization/rule contributes,  
2. **Learned costs** (or at least confusion-matrix validation) to demonstrate costs are not arbitrary, and  
3. **Human-judgment correlation** to show your metric tracks perceived correctness in Gulf poetic contexts (meaning, rhyme, and meter) rather than just orthographic similarity. citeturn18view0turn17view0turn12search16turn19search0