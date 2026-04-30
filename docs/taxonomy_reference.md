# Nabati Poetry Taxonomy Reference

**Author:** Asma Salem Mubarak Najem Aljneibi
**Version:** 1.0 | **Date:** 2026-03-21
**Scope:** Standalone research glossary for Khaleeji/Nabati poetry — genre and emotion. Intended for standardisation and reuse across projects.

---

## Overview

This document is a comprehensive, research-based reference for the **genre** and **emotion** taxonomies of Khaleeji/Nabati poetry. It presents all classes in their full, unmerged form, maps each class against established reference taxonomies from the literature, and documents the cultural and linguistic grounding for every label. It does not make project-specific implementation decisions (class merging for model training, label encoding, etc.) — those belong in implementation documentation.

---

## Part I — Genre Taxonomy (11 Classes)

### 1.1 Background

Nabati (Khaleeji/Gulf) poetry is a living oral literary tradition of the Arabian Peninsula. It is composed in Gulf vernacular Arabic rather than Classical Modern Standard Arabic (MSA) and has its own genre system that partially overlaps with, but substantially diverges from, the Classical Arabic *aghrad* (أغراض) tradition.

The primary authoritative cultural reference is:

> **Holes, C. and Abu Athera, S. S.** *The Nabati Poetry of the United Arab Emirates.* Ithaca Press, Reading, 2011. ISBN 978-0-86372-383-4.

Classical Arabic poetics references used for alignment:

> **Al-Jurjani, Abd al-Qadir.** *Asrar al-Balagha.* Dar al-Ma'arif, Cairo, 1959.
> **Qudama ibn Ja'far.** *Naqd al-Shi'r.* 9th century CE.

### 1.2 The 11 Nabati Genre Classes

Each class is defined with its English label, Arabic label, transliteration, core thematic domain, characteristic linguistic features, and typical emotional register.

---

#### Genre 1 — Ghazal (Delicate Love)
**Arabic:** غزل (رقيق / حب رقيق) | **Transliteration:** *Ghazal*

**Definition:** Romantic love poetry characterised by tender address to a beloved, delicate emotional register, longing in the beloved's absence, and intimate complaint. Nabati Ghazal uses Gulf vernacular vocabulary and differs from Classical Ghazal in its more personal, less elevated diction.

**Core themes:** Romantic longing, tender address to beloved, delicate love, intimate complaint, the pain of separation, the beauty of the beloved.

**Linguistic markers:** Second-person address (*ya habib*, يا حبيب), diminutive and softening forms, high frequency of negation particles, love-pain register vocabulary.

**Typical emotion:** Delicate Love, Longing, Sorrow, Hope, Disappointment.

**Classical Arabic equivalent:** غزل (*Ghazal*) — direct continuity; Nabati Ghazal uses Gulf vernacular morphology.

---

#### Genre 2 — Shajan (Sorrow / Regret)
**Arabic:** شجن (حزن وأسى) | **Transliteration:** *Shajan*

**Definition:** A distinctly Nabati lyric genre centred on grief, deep sorrow, and regret. More inward and personally grievous than Ritha (which mourns a specific person). Shajan poems perform emotional weight as aesthetic value in itself.

**Core themes:** Grief, longing, regret, emotional weight of loss, the experience of sorrow as a state of being.

**Linguistic markers:** Grief-register vocabulary (*shajani*, *yuhzunni*), first-person introspective voice, high density of negative particles and lament constructions.

**Typical emotion:** Sorrow, Longing, Disappointment, Contemplation.

**Classical Arabic equivalent:** No direct classical equivalent. Closest to the *nasib* (ناسيب) — the elegiac prelude in classical *qasida* — but Shajan is a standalone genre, not a section of a larger poem.

---

#### Genre 3 — Fakhr (Pride & Honor)
**Arabic:** فخر (عزة وشموخ) | **Transliteration:** *Fakhr*

**Definition:** Poetry of self-assertion, tribal or personal pride, boasting of lineage, courage, and honour. In Gulf oral tradition Fakhr includes both self-directed boasting and public affirmation of a group's status. It is a performative genre — the poet enacts pride as a social act.

**Core themes:** Self-assertion, tribal honour, courage, lineage pride, public validation of status.

**Linguistic markers:** First-person assertive constructions, lineage references, martial and honour vocabulary (*izza*, *shamon*, *karama*).

**Typical emotion:** Pride, Defiance, Admiration.

**Classical Arabic equivalent:** فخر (*Fakhr*) — direct continuity across classical and Nabati traditions.

---

#### Genre 4 — Hikma (Wisdom, Philosophical & Reflection)
**Arabic:** حكمة (فلسفة وتأمل) | **Transliteration:** *Hikma*

**Definition:** Poetry of philosophical reflection, moral guidance, life wisdom, and religious contemplation. Encompasses advisory verse, ascetic poetry, and meditative observation. In the Nabati tradition Hikma often incorporates Islamic references and draws on Bedouin ecological observation as a source of wisdom.

**Core themes:** Moral guidance, philosophical contemplation, religious reflection, life advice, the passage of time, observation of nature as a source of wisdom.

**Linguistic markers:** Gnomic present tense, impersonal constructions, proverbial register, wisdom-vocabulary (*hikma*, *'aql*, *sabr*).

**Typical emotion:** Contemplation, Hope, Admiration, Pride.

**Classical Arabic equivalent:** حكمة (*Hikma*) / زهد (*Zuhd* — asceticism) — direct alignment. Religious and advisory poetry unified under this category.

---

#### Genre 5 — Badawa (Bedouin Life & Desert Heritage)
**Arabic:** بداوة (حياة البدو والبر) | **Transliteration:** *Badawa*

**Definition:** Poetry celebrating and documenting Bedouin desert life, the natural world of the Gulf, traditional practices (falconry, camel husbandry, desert travel), and heritage. A uniquely Khaleeji genre with no classical equivalent. Badawa poetry functions partly as an oral archive of traditional knowledge.

**Core themes:** Desert landscape, falconry, camels, traditional Bedouin life, natural world, heritage and memory, oral documentation of practices.

**Linguistic markers:** Nature vocabulary (desert, falcons, camels, plants, stars), technical Bedouin terminology, descriptive and narrative voice.

**Typical emotion:** Contemplation, Pride, Neutral / Descriptive, Admiration, Longing.

**Classical Arabic equivalent:** وصف (*Wasf* — description) in part; no direct equivalent for the Bedouin-life thematic cluster.

---

#### Genre 6 — Wataniyya (Patriotic & National)
**Arabic:** وطنية (حب الوطن والانتماء) | **Transliteration:** *Wataniyya*

**Definition:** Patriotic and nationalist poetry celebrating homeland, national identity, belonging, and loyalty to the state or ruler. A predominantly modern genre in Gulf poetry, emerging in the 20th century alongside the formation of Gulf states, though rooted in earlier tribal loyalty verse.

**Core themes:** Homeland love, national identity, patriotic celebration, belonging, loyalty, national occasions.

**Linguistic markers:** Homeland vocabulary (*watan*, *ardh*, *sha'b*), national symbols, second-person address to the homeland, inclusive first-person plural (*nihna*, *nahnu*).

**Typical emotion:** Pride, Hope, Admiration, Defiance.

**Classical Arabic equivalent:** No direct classical equivalent; closest to *madih* of the ruler in function, though now directed at the nation-state rather than an individual patron.

---

#### Genre 7 — Ritha (Elegy & Lament)
**Arabic:** رثاء (ندب وعزاء) | **Transliteration:** *Ritha*

**Definition:** Elegiac poetry mourning a specific deceased person — a family member, tribal leader, poet, or public figure. Ritha praises the deceased, expresses the grief of those left behind, and often provides comfort to the bereaved. Distinct from Shajan (general sorrow) in its specific addressee.

**Core themes:** Mourning a specific person, eulogy, praise of the deceased, comfort of the bereaved, memory of the departed, the finality of death.

**Linguistic markers:** Proper names and honorifics, third-person reference to the deceased, grief vocabulary (*rahil*, *wada'*, *ghayyab*), direct address to the deceased.

**Typical emotion:** Sorrow, Longing, Compassion, Admiration.

**Classical Arabic equivalent:** رثاء (*Ritha*) — direct continuity across all periods of Arabic poetry.

---

#### Genre 8 — Hija (Satire & Social Critique)
**Arabic:** هجاء (نقد وسخرية) | **Transliteration:** *Hija*

**Definition:** Satirical and critical poetry targeting individuals, social behaviours, or cultural changes. Ranges from sharp personal attack to broader social observation and ironic commentary on community norms. In Nabati tradition Hija is a performative act with social consequences — public shaming through verse.

**Core themes:** Satirical critique, social observation, humour and irony, verbal attack, social norms, cultural commentary.

**Linguistic markers:** Ironic register, naming and shaming constructions, exaggeration, diminutive and belittling forms, rhetorical questions.

**Typical emotion:** Defiance, Humor, Sorrow, Disappointment.

**Classical Arabic equivalent:** هجاء (*Hija*) — direct continuity; one of the oldest Arabic genres.

---

#### Genre 9 — Madih (Praise)
**Arabic:** مديح | **Transliteration:** *Madih*

**Definition:** Poetry of praise directed at a specific person — a ruler, tribal leader, benefactor, or admired figure. Madih enacts social exchange: the poet confers public honour upon the subject. It differs from Fakhr (self-praise) in that the locus of pride is the praised person rather than the poet's own lineage.

**Core themes:** Public praise of a person, conferral of honour, celebration of virtues, social exchange between poet and patron.

**Linguistic markers:** Second- or third-person laudatory constructions, honoured titles and epithets, elevation vocabulary.

**Typical emotion:** Admiration, Pride (in the subject), Hope.

**Classical Arabic equivalent:** مديح (*Madih*) — direct continuity; arguably the most central genre in Classical Arabic poetry, associated with the qasida structure.

---

#### Genre 10 — I'tithar (Delicate Apology)
**Arabic:** اعتذار رقيق | **Transliteration:** *Iʿtithār*

**Definition:** A tender poetry of apology or reconciliation addressed to a beloved or a person the poet has wronged. The apology is framed with affection and delicacy; the register is intimate and private, closely related to the love-poetry register of Ghazal. Unique to the Nabati tradition as a named standalone genre.

**Core themes:** Tender apology, reconciliation, appeal for forgiveness, intimate acknowledgement of fault, preservation of the relationship.

**Linguistic markers:** Softening and diminutive forms, conditional constructions (*law*, *in*), first-person confession, second-person appeal, love-register vocabulary.

**Typical emotion:** Delicate Love, Longing, Hope, Disappointment, Sorrow.

**Classical Arabic equivalent:** No direct equivalent as a standalone genre; elements appear as subthemes within Ghazal or reconciliation sections of longer poems.

---

#### Genre 11 — Tareef (Humorous)
**Arabic:** تعريف / طرفة | **Transliteration:** *Tārīf / Ṭurfa*

**Definition:** Light humorous verse characterised by wit, wordplay, and social observation. Less cutting than Hija; the intent is amusement rather than attack. Tareef relies on shared cultural knowledge, situational irony, and comic exaggeration. Functions as entertainment and social lubricant in performance contexts.

**Core themes:** Wit, wordplay, social observation through a comic lens, amusing anecdotes, playful exaggeration.

**Linguistic markers:** Comic register, playful exaggeration, unexpected juxtapositions, punning and wordplay.

**Typical emotion:** Humor, Neutral / Descriptive, Admiration.

**Classical Arabic equivalent:** Closest to الظرف (*al-Zurf* — wit and elegance) or humorous subgenres of *adab* literature; no direct classical genre equivalent.

---

### 1.3 Alignment with Holes & Abu Athera (2011)

| Volume Section | Nabati Genre (EN) | Arabic Label | Rationale |
|----------------|------------------|-------------|-----------|
| Section 1: Boasting Poems | **Fakhr** | فخر | Direct match |
| Section 2: Patriotic Poems | **Wataniyya** | وطنية | Direct match |
| Section 3: Religious Poems | **Hikma** | حكمة | Religious poetry's functions overlap with contemplation and moral guidance |
| Section 4: The Skill of the Poet / Meta-Poetry | **Fakhr** | فخر | Poetic self-assertion is a form of Fakhr |
| Section 5: Hunting Poems | **Badawa** | بداوة | Falconry and hunting are embedded in Bedouin desert culture |
| Section 6: Dialogue Poems & Correspondence | **Badawa** (default) | بداوة | Multi-functional; Badawa is the nearest cultural register as a default |
| Section 7: Love Poems | **Ghazal** | غزل | Direct match |
| Section 8: Social Poems | **Hija** | هجاء | Social critique and commentary |
| Section 9: Poems of Advice & Guidance | **Hikma** | حكمة | Direct match |
| Section 10: Elegies | **Ritha** | رثاء | Direct match |

### 1.4 Alignment with Classical Arabic Genre Tradition (*Aghrad*)

| Classical Genre (AR) | Classical Genre (EN) | Nabati Equivalent | Notes |
|---------------------|---------------------|------------------|-------|
| غزل (*Ghazal*) | Romantic love | **Ghazal** | Direct continuity; vernacular vocabulary |
| فخر (*Fakhr*) | Self-boasting / tribal pride | **Fakhr** | Direct continuity |
| مديح (*Madih*) | Praise of patron | **Madih** | Direct continuity; now a separate named genre |
| هجاء (*Hija*) | Satirical invective | **Hija** | Direct continuity |
| رثاء (*Ritha*) | Elegy / lament | **Ritha** | Direct continuity |
| وصف (*Wasf*) | Description of nature / objects | **Badawa** (dominant) | Descriptive verse in Nabati context is anchored to desert heritage |
| حكمة / زهد (*Hikma / Zuhd*) | Wisdom / asceticism | **Hikma** | Direct continuity |
| ناسيب (*Nasib*) | Elegiac prelude | **Shajan** (closest) | Shajan elevates the nasib's emotional register into a standalone genre |
| — | — | **Wataniyya** | Modern development; no classical equivalent |
| — | — | **Badawa** | Uniquely Khaleeji; no classical equivalent |
| — | — | **I'tithar** | No classical equivalent as standalone genre |
| — | — | **Tareef** | No direct classical genre equivalent |

---

## Part II — Emotion Taxonomy (12 Classes)

### 2.1 Background and Motivation

Standard emotion taxonomies — Ekman's six basic emotions, Plutchik's wheel, the NRC Lexicon — were developed on English-language data and do not capture the aesthetic emotion categories central to performed Arabic poetry. Three specific mismatches motivate a domain-specific taxonomy:

1. **Aesthetic vs. utilitarian emotions:** Poetry foregrounds *Contemplation* (Ta'ammul) and *Delicate Love* (Hub Raqeeq) as primary categories; Ekman's taxonomy has no equivalent.
2. **Delivery mismatch:** In Nabati oral performance, a grief poem delivered with vigorous vocal energy is an intentional artistic device. Arousal and valence are regularly decoupled; standard circumplex models do not account for this.
3. **Register-specific emotions:** *Defiance* (Tahaddi), *Longing* (Shawq), and *Compassion* (Hanaan) are culturally central and absent from generic Western taxonomies.

The closest external reference is the PO-EMO aesthetic emotion taxonomy for poetry (Haider et al., LREC 2020), though it was developed for German and English verse.

### 2.2 The 12 Nabati Emotion Classes

Each class is defined with its English label, Arabic label, transliteration, affective description, characteristic linguistic and performative cues, valence–arousal position, and cross-reference to external taxonomies.

---

#### Emotion 1 — Longing (Shawq)
**Arabic:** شوق | **Transliteration:** *Shawq*

**Definition:** The painful desire for an absent beloved, place, or time. Shawq in Nabati poetry is not passive nostalgia but an active, aching state. It co-occurs with grief but is directional — directed toward something desired and absent.

**Affective profile:** Valence: negative-mixed (the desire itself carries beauty; the absence carries pain). Arousal: Low–Medium.

**Performative cues:** Extended melodic phrases, drawn-out vowels (إشباع *ish'ba*'), quiet intensity.

**Linguistic markers:** Lexical field of absence (*ba'eed*, *ghayib*, *wissal*), longing-directed constructions (*ishtaqt ilak*).

**PO-EMO parallel:** Nostalgia (strong alignment). **Ekman:** Sadness (partial). **Plutchik:** Sadness + Anticipation. **Russell:** Negative-mixed, Low.

---

#### Emotion 2 — Delicate Love (Hub Raqeeq)
**Arabic:** حب رقيق | **Transliteration:** *Ḥub Raqīq*

**Definition:** Tender, gentle, refined love — affection marked by delicacy rather than passion. The Nabati tradition distinguishes this from intense or erotic love; it is closer to devotion, adoration, and gentle care.

**Affective profile:** Valence: Positive. Arousal: Low.

**Performative cues:** Soft vocal delivery, intimate diction, slow tempo.

**Linguistic markers:** Diminutives and softening forms, tender address (*ya ruh*, *ya 'ein*), gentle register vocabulary.

**PO-EMO parallel:** Beauty / Joy (partial). **Ekman:** Happiness (low intensity). **Plutchik:** Love (Joy + Trust). **Russell:** Positive, Low.

---

#### Emotion 3 — Sorrow (Huzn)
**Arabic:** حزن | **Transliteration:** *Ḥuzn*

**Definition:** Deep sadness and grief as an emotional state. Unlike Shawq (directed at an absence), Huzn is a more general state of inner heaviness. It is one of the most celebrated states in Nabati oral performance — the ability to move an audience to grief is a marker of poetic mastery.

**Affective profile:** Valence: Negative. Arousal: Low–Medium.

**Performative cues:** Plaintive vocal quality, deliberate pace, collective audience response (*tarab* — emotional resonance).

**Linguistic markers:** Grief vocabulary (*huzn*, *kamd*, *alam*), heavy imagery (night, exile, stone).

**PO-EMO parallel:** Sadness (strong alignment). **Ekman:** Sadness. **Plutchik:** Sadness. **Russell:** Negative, Low.

---

#### Emotion 4 — Pride (Fakhr)
**Arabic:** فخر | **Transliteration:** *Fakhr*

**Definition:** Strong positive self-assertion rooted in honour, lineage, and personal or tribal achievement. As an emotion, Pride is the affective dimension of the Fakhr genre — though it also appears in other genres (Wataniyya, Badawa). It is high-energy and public-facing.

**Affective profile:** Valence: Positive. Arousal: High.

**Performative cues:** Forceful delivery, heightened pace, strong rhythmic emphasis, projective vocal quality.

**Linguistic markers:** First-person assertive constructions, honour vocabulary (*izza*, *karama*, *majd*), lineage references.

**PO-EMO parallel:** Vitality (moderate). **Ekman:** No direct equivalent (closest: Happiness at high arousal). **Plutchik:** Joy (high intensity). **Russell:** Positive, High.

---

#### Emotion 5 — Admiration (I'jab)
**Arabic:** إعجاب | **Transliteration:** *Iʿjāb*

**Definition:** Appreciation and admiration directed toward a person, quality, landscape, or achievement. Less assertive than Pride; the gaze is outward rather than self-referential. It carries warmth and recognition.

**Affective profile:** Valence: Positive. Arousal: Medium.

**Performative cues:** Appreciative, often measured delivery; celebratory but not forceful.

**Linguistic markers:** Laudatory constructions, beauty and virtue vocabulary, third-person praise register.

**PO-EMO parallel:** Beauty / Joy (moderate). **Ekman:** Happiness. **Plutchik:** Trust + Joy. **Russell:** Positive, Medium.

---

#### Emotion 6 — Contemplation (Ta'ammul)
**Arabic:** تأمل | **Transliteration:** *Taʾammul*

**Definition:** Philosophical or meditative reflection — the emotion of deep thinking about life, nature, time, or meaning. Ta'ammul is a distinctly Nabati-poetry category absent from Western emotion taxonomies. It is neither positive nor negative but carries aesthetic value in its own right.

**Affective profile:** Valence: Neutral. Arousal: Low.

**Performative cues:** Slow, measured pace; minimal ornamentation; quiet delivery that invites the audience into reflection.

**Linguistic markers:** Gnomic constructions, philosophical register, time and nature imagery, generalising statements.

**PO-EMO parallel:** No direct equivalent — **Nabati-unique**. Closest: a reflective blend of Nostalgia and Suspense/Expectation, but neither captures the purely contemplative quality.

**Ekman:** No equivalent. **Plutchik:** No equivalent. **Russell:** Neutral, Low.

---

#### Emotion 7 — Disappointment (Khayba)
**Arabic:** خيبة | **Transliteration:** *Khayba*

**Definition:** The deflation and heaviness of thwarted expectation — hope that did not materialise, trust that was betrayed, or a desired outcome that failed. Khayba is more specific than general sorrow; it always refers to a gap between expectation and reality.

**Affective profile:** Valence: Negative. Arousal: Low.

**Performative cues:** Subdued, resigned delivery; heavy pausing.

**Linguistic markers:** Counterfactual constructions (*laytani*, *kuntu atawaqqa'*), failed-expectation vocabulary, betrayal register.

**PO-EMO parallel:** Uneasiness (partial). **Ekman:** Sadness (partial). **Plutchik:** Sadness. **Russell:** Negative, Low.

---

#### Emotion 8 — Defiance (Tahaddi)
**Arabic:** تحدي | **Transliteration:** *Taḥaddī*

**Definition:** Bold, assertive challenge directed at an adversary, a difficult situation, or fate itself. Defiance is positive-assertive rather than angry; it channels strength and determination. In the Nabati tradition Tahaddi is a celebrated performance emotion — the poet's verbal courage is itself an act of honour.

**Affective profile:** Valence: Positive-assertive. Arousal: High.

**Performative cues:** Forceful, projected delivery; sharp rhythmic attack; direct address to the challenged party.

**Linguistic markers:** Challenge constructions (*ana*, *mani khayif*), adversarial address, strength and bravery vocabulary.

**PO-EMO parallel:** Tension (partial — PO-EMO's Tension is anxious; Tahaddi is assertive). **Ekman:** Anger (sublimated into assertion). **Plutchik:** Anger + Anticipation. **Russell:** Positive-assertive, High.

---

#### Emotion 9 — Hope (Amal)
**Arabic:** أمل | **Transliteration:** *Amal*

**Definition:** Forward-looking aspiration, anticipation of a positive future, or the persistence of belief in possibility despite present difficulty. In Nabati poetry Hope frequently co-occurs with Sorrow — the poem holds both simultaneously, and this tension is aesthetically valued.

**Affective profile:** Valence: Positive. Arousal: Medium.

**Performative cues:** Lighter delivery than grief poems; upward melodic contours.

**Linguistic markers:** Future constructions, aspiration vocabulary (*amal*, *tawaqqu'*, *bukra*), conditional hope constructions.

**PO-EMO parallel:** Suspense / Expectation (partial). **Ekman:** Happiness (anticipatory). **Plutchik:** Anticipation. **Russell:** Positive, Medium.

---

#### Emotion 10 — Compassion (Hanaan)
**Arabic:** حنان | **Transliteration:** *Ḥanān*

**Definition:** Warm, tender care and empathy — a gentle outward-directed emotion of nurturing and protectiveness. Hanaan is most prominent in Ritha (elegy) and in poems addressed to the vulnerable. It is distinct from love (which is dyadic and romantic) in being unidirectional and nurturing.

**Affective profile:** Valence: Positive-warm. Arousal: Low.

**Performative cues:** Soft, protective vocal quality; gentle pace; nurturing register.

**Linguistic markers:** Softening forms, protective constructions, care vocabulary (*hanaan*, *raha*, *hilm*).

**PO-EMO parallel:** No close equivalent; elements of Beauty / Joy. **Ekman:** Happiness (nurturing variant). **Plutchik:** Trust. **Russell:** Positive, Low.

---

#### Emotion 11 — Humor (Turfah)
**Arabic:** طرفة | **Transliteration:** *Ṭurfa*

**Definition:** Light wit and amusement generated through wordplay, irony, comic observation, or playful exaggeration. Turfah in Nabati poetry is social and performative — it creates communal pleasure and signals the poet's verbal dexterity. It is lighter than Hija (satire) with no intent to harm.

**Affective profile:** Valence: Positive-light. Arousal: Medium.

**Performative cues:** Quick pace, playful intonation, audience laughter response, comic timing.

**Linguistic markers:** Comic register, wordplay, unexpected juxtaposition, diminutive and exaggerating forms.

**PO-EMO parallel:** Amusement (moderate). **Ekman:** Happiness. **Plutchik:** Joy. **Russell:** Positive, Medium.

---

#### Emotion 12 — Neutral / Descriptive (Wasfi)
**Arabic:** وصفي / محايد | **Transliteration:** *Waṣfī*

**Definition:** Descriptive or observational narration where no dominant affective state is foregrounded. The poem documents, describes, or narrates without strong emotional colouring. This is not emotional flatness — it reflects the oral tradition's documentary and archival function.

**Affective profile:** Valence: Neutral. Arousal: Low.

**Performative cues:** Even delivery, measured pace, informational register, minimal ornamentation.

**Linguistic markers:** Third-person or generalised narration, descriptive vocabulary, low frequency of emotional intensifiers.

**PO-EMO parallel:** No direct equivalent — **Nabati-unique**. The oral tradition's documentary function produces this category, which has no counterpart in Western poetry emotion frameworks.

**Ekman:** No equivalent. **Plutchik:** No equivalent. **Russell:** Neutral, Low.

---

### 2.3 Valence–Arousal Map (Russell's Circumplex)

Positions are approximate; Nabati emotion categories do not map cleanly onto a purely dimensional space.

```
                       HIGH AROUSAL
                            |
          Defiance       Pride
          (Tahaddi)      (Fakhr)
                \           |
                 \    Admiration (I'jab)
                  \    /
NEGATIVE ──────────+──────────── POSITIVE
                  / \
      Disappointment  Hope (Amal)
         (Khayba)       \
                          Delicate Love (Hub Raqeeq)
       Sorrow (Huzn)      Compassion (Hanaan)
              |
        Longing (Shawq)
              |
       Contemplation (Ta'ammul)    Neutral (Wasfi)
                            |
                       LOW AROUSAL
```

**Key observation — Delivery Mismatch:** In Nabati oral performance, vocal arousal deliberately contradicts textual emotion in over half of all clips. A grief poem (low arousal by text) is commonly performed with high vocal energy. This is a culturally significant expressive device, not annotation noise. Standard circumplex models do not account for this decoupling.

### 2.4 Alignment with External Emotion Taxonomies

#### Ekman's Basic Emotions (1992)

| Ekman Emotion | Nabati Manifestation | Notes |
|--------------|----------------------|-------|
| Happiness | Delicate Love, Hope, Admiration, Humor | Nabati positive affect is finely differentiated; undifferentiated "happiness" is not a primary category |
| Sadness | Sorrow, Longing, Disappointment | Good correspondence; Nabati further differentiates by directionality and depth |
| Anger | Defiance (sublimated form) | Pure anger is not a celebrated Nabati poetic affect; it is ritually channelled into assertive Defiance (Tahaddi). The tradition valorises controlled strength over raw rage |
| **Fear** | Embedded in Longing, Sorrow, and negated in Defiance/Pride | Fear (*khawf*, خوف) is not a standalone Nabati emotion class but is culturally present in two distinct forms: (1) **as subtext** — fear of separation and abandonment animates Longing (Shawq) and Ghazal; fear of loss runs beneath Sorrow (Huzn) and Ritha; fear of fate and death surfaces within Contemplation in Hikma poetry; (2) **as negation** — the performative declaration "I am not afraid" (*mani khayif*, ما أني خايف) is one of the most recognisable Fakhr and Defiance constructions. The tradition does not celebrate fear as an aesthetic state but rather addresses it by publicly overcoming it |
| **Disgust** | Expressed through Hija genre; social register rather than personal affective state | Disgust (*istinkar*, استنكار; *karahiyya*, كراهية) is the animating moral affect behind Hija (Satire & Social Critique). In Nabati tradition it is externalised and ritualised — the poet performs social disgust as a public act with collective consequences. Because it is genre-bound and outward-directed rather than a private interior state, it does not form a standalone emotion class; instead it is encoded at the genre level |
| **Surprise** | Embedded in Admiration and Contemplation | Wonder and astonishment (*'ajab*, عجب; *dasha*, دهشة) are not standalone Nabati emotion classes but surface as a sub-register within two categories: (1) **Admiration (I'jab)** — the word itself derives from the Arabic root for wonder (*'ajaba*, أعجب), and Admiration in Nabati poetry carries an element of struck amazement at beauty or virtue; (2) **Contemplation (Ta'ammul)** — encountering the sublime in nature (desert vastness, the night sky, the falcon) provokes a form of awed reflection that blends surprise with meditation |
| — | **Pride** | Nabati-unique primary category; no Ekman equivalent |
| — | **Contemplation** | Nabati-unique; philosophical reflection as a primary aesthetic emotion has no Ekman counterpart |
| — | **Compassion (Hanaan)** | Approximated broadly by Happiness in Ekman; functionally distinct in Nabati as outward-directed nurturing care |
| — | **Neutral / Descriptive** | Nabati-unique; the oral tradition's documentary register produces this as a primary category |

> **Reference:** Ekman, P. "An argument for basic emotions." *Cognition and Emotion* 6(3–4), 169–200, 1992.

#### Plutchik's Wheel of Emotions (1980)

| Plutchik Emotion | Nabati Manifestation | Notes |
|-----------------|----------------------|-------|
| Joy | Hope, Delicate Love (low intensity), Humor | Nabati positive states are lower arousal and more finely differentiated than Plutchik's undifferentiated Joy |
| Trust | Admiration, Compassion | Relational and appreciative frame; Compassion (Hanaan) is the nurturing form, Admiration (I'jab) the appreciative form |
| **Fear** | Embedded as subtext in Longing, Sorrow; negated in Pride and Defiance | In Plutchik's model Fear ranges from apprehension to terror. In Nabati poetry, *khawf* (خوف) exists but is not performed as a primary state: at low intensity it underlies the anxiety of separation in Longing (Shawq) and the dread of permanent loss in Sorrow (Huzn) and Ritha; at higher intensity it is invoked only to be publicly defeated in Fakhr and Defiance contexts. The tradition's moral economy does not reward fear as a performed aesthetic; it rewards its conquest |
| **Surprise** | Sub-register of Admiration and Contemplation | Plutchik's Surprise (ranging from distraction to amazement) maps partially onto the wonder component of Admiration (I'jab, etymologically rooted in *'ajab*) and the awed dimension of Contemplation when facing the sublime in nature. Neither forms a standalone class in Nabati |
| Sadness | Sorrow, Longing, Disappointment | Strong alignment; Nabati differentiates by cause: loss of person → Sorrow; loss of presence → Longing; loss of expectation → Disappointment |
| **Disgust** | Social moral function of Hija genre | Plutchik's Disgust (boredom to loathing) is the moral engine of Hija poetry. In Nabati performance tradition, disgust is ritualised and directed outward toward social actors or behaviours. Because it operates at the genre level — shaping the social act of satirical verse — rather than as a private internal state, it is not isolated as a standalone emotion class |
| Anger | Defiance (sublimated) | Plutchik's Anger (annoyance to rage) is channelled into assertive challenge in Nabati; the tradition valorises controlled verbal force over uncontrolled anger |
| Anticipation | Hope | Close alignment; Hope (Amal) carries the forward-looking expectation that Plutchik defines as Anticipation |
| *Love* (complex) | Delicate Love | Close alignment; Nabati refines this further into tender, non-erotic register |
| *Optimism* (complex: Joy + Anticipation) | Hope + Pride | — |
| *Remorse* (complex: Sadness + Disgust) | Disappointment + Sorrow | — |
| *Awe* (complex: Fear + Surprise) | Contemplation (partial) | The awe-struck dimension of Contemplation in Hikma and Badawa poetry partially captures this complex |

> **Reference:** Plutchik, R. *Emotion: A Psychoevolutionary Synthesis.* Harper & Row, 1980.

#### PO-EMO Aesthetic Poetry Emotions (Haider et al., 2020)

PO-EMO is the most relevant external reference, having been developed specifically for poetry.

| PO-EMO Category | Nabati Equivalent(s) | Correspondence |
|----------------|----------------------|---------------|
| Uneasiness | Disappointment, Sorrow (partial) | Moderate |
| Tension | Defiance (partial) | Weak — PO-EMO Tension is anxious; Nabati Defiance is assertive |
| Sadness | Sorrow, Disappointment | Strong |
| Nostalgia | Longing (Shawq) | Strong |
| Amusement | Humor (Turfah) | Moderate |
| Beauty / Joy | Delicate Love, Hope, Admiration | Moderate |
| Vitality | Pride (Fakhr), Defiance (Tahaddi) | Moderate |
| Suspense / Expectation | Hope (Amal) | Weak |
| *No PO-EMO equivalent* | **Contemplation (Ta'ammul)** | **Nabati-unique** |
| *No PO-EMO equivalent* | **Neutral / Descriptive (Wasfi)** | **Nabati-unique** |
| *No PO-EMO equivalent* | **Compassion (Hanaan)** | Distinct in Nabati; not captured in PO-EMO |

> **Reference:** Haider, M., Kim, E., Kuhn, G., and Zweig, G. "PO-EMO: Conceptualization, Annotation, and Modeling of Aesthetic Emotions in German and English Poetry." *Proceedings of LREC 2020*, pp. 1652–1663. [ACL Anthology](https://aclanthology.org/2020.lrec-1.205/)

#### NRC Emotion Lexicon (Mohammad & Turney, 2013)

| NRC Emotion | Nabati Manifestation | Notes |
|------------|---------------------|-------|
| Joy | Delicate Love, Hope, Admiration, Humor | Differentiated into four distinct positive classes in Nabati |
| Trust | Admiration, Compassion | Outward appreciation (Admiration) and nurturing care (Compassion) |
| **Fear** | Subtext in Longing, Sorrow; negated in Pride and Defiance | *Khawf* (خوف) underlies the anxiety of separation in Longing and Ghazal, and the dread of loss in Sorrow and Ritha; simultaneously, its explicit negation ("I am not afraid") is a defining linguistic marker of Fakhr and Defiance. Fear is present in the tradition but expressed through these secondary positions and negations rather than as a foregrounded primary class |
| **Surprise** | Sub-register within Admiration and Contemplation | Wonder (*'ajab*, عجب) is etymologically embedded in Admiration (I'jab) and surfaces in Contemplation when encountering the sublime in desert nature or the vastness of time. Not a standalone class |
| Sadness | Sorrow, Longing, Disappointment | Strong three-way differentiation by cause and directionality |
| **Disgust** | Animating affect of Hija genre | Social moral disgust (*istinkar*, استنكار) drives satirical verse and is encoded at genre level rather than as a named internal emotion class. The NRC category is present in the tradition but performed publicly and structurally through Hija rather than privately labelled |
| Anger | Defiance (sublimated) | Channelled into controlled verbal assertion; raw anger is not a celebrated Nabati performative state |
| Anticipation | Hope (Amal) | Close alignment |

> **Reference:** Mohammad, S. M. and Turney, P. D. "Crowdsourcing a Word-Emotion Association Lexicon." *Computational Intelligence* 29(3), 436–465, 2013.

---

## Part III — Genre–Emotion Co-occurrence Map

This table documents which emotions are culturally and statistically plausible for each genre. It is grounded in both corpus statistics and Nabati tradition knowledge and is useful for genre-constrained emotion analysis.

| Genre | Primary Emotions | Secondary Emotions |
|-------|-----------------|-------------------|
| **Ghazal** | Delicate Love, Longing | Sorrow, Hope, Disappointment |
| **Shajan** | Sorrow, Longing | Disappointment, Contemplation |
| **Fakhr** | Pride, Defiance | Admiration |
| **Hikma** | Contemplation, Hope | Admiration, Pride |
| **Badawa** | Contemplation, Pride | Neutral / Descriptive, Admiration, Longing |
| **Wataniyya** | Pride, Hope | Admiration, Defiance |
| **Ritha** | Sorrow, Longing | Compassion, Admiration |
| **Hija** | Defiance, Humor | Sorrow, Disappointment |
| **Madih** | Admiration, Pride | Hope |
| **I'tithar** | Delicate Love, Longing | Hope, Disappointment, Sorrow |
| **Tareef** | Humor, Neutral / Descriptive | Admiration |

---

## Part IV — Arousal Taxonomy (3 Classes)

Arousal is assessed from acoustic features of the audio performance, independently of the textual emotion category.

| ID | Label | Arabic | Definition |
|----|-------|--------|-----------|
| 0 | **High** | عالٍ | Energetic, forceful delivery; loud, fast, rhythmically intense |
| 1 | **Low** | منخفض | Quiet, slow, subdued delivery; intimate or meditative |
| 2 | **Neutral** | محايد | Neither distinctly high nor low; balanced delivery |

### Emotion-to-Arousal Expected Mapping

| Emotion | Expected Arousal |
|---------|:----------------:|
| Pride (Fakhr) | High |
| Defiance (Tahaddi) | High |
| Admiration (I'jab) | Medium–High |
| Hope (Amal) | Medium |
| Humor (Turfah) | Medium |
| Longing (Shawq) | Low |
| Sorrow (Huzn) | Low |
| Disappointment (Khayba) | Low |
| Contemplation (Ta'ammul) | Low |
| Delicate Love (Hub Raqeeq) | Low |
| Compassion (Hanaan) | Low |
| Neutral / Descriptive (Wasfi) | Low |

### Delivery Mismatch

**Delivery Mismatch** occurs when the expected arousal of the textual emotion contradicts the predicted acoustic arousal. In Nabati oral tradition this is a culturally significant expressive device — performing grief poetry with high vocal energy is intentional and artistically recognised, not an error.

---

## Part V — Consolidated Cross-Reference Table

### 5.1 Nabati → External Systems (by Nabati class)

| Nabati Class (EN) | Arabic | Ekman | Plutchik | Russell (V, A) | PO-EMO | NRC |
|-------------------|--------|-------|----------|---------------|--------|-----|
| Longing (Shawq) | شوق | Sadness | Sadness + Anticipation | Neg-mixed, Low | Nostalgia | Sadness |
| Delicate Love (Hub Raqeeq) | حب رقيق | Happiness (low) | Love (Joy + Trust) | Pos, Low | Beauty/Joy | Joy, Trust |
| Sorrow (Huzn) | حزن | Sadness | Sadness | Neg, Low | Sadness | Sadness |
| Pride (Fakhr) | فخر | *Nabati-unique primary class* | Joy (high intensity) | Pos, High | Vitality | Joy |
| Admiration (I'jab) | إعجاب | Happiness | Trust + Joy | Pos, Med | Beauty/Joy | Trust, Joy |
| Contemplation (Ta'ammul) | تأمل | *Nabati-unique — no Ekman equivalent* | *Nabati-unique — no Plutchik equivalent* | Neutral, Low | *Nabati-unique* | *Nabati-unique* |
| Disappointment (Khayba) | خيبة | Sadness | Sadness | Neg, Low | Uneasiness | Sadness |
| Defiance (Tahaddi) | تحدي | Anger (sublimated) | Anger + Anticipation | Pos-assertive, High | Tension (partial) | Anger |
| Hope (Amal) | أمل | Happiness (anticipatory) | Anticipation | Pos, Med | Suspense/Expectation | Anticipation, Joy |
| Compassion (Hanaan) | حنان | Happiness (nurturing) | Trust | Pos, Low | *No close equivalent* | Trust |
| Humor (Turfah) | طرفة | Happiness | Joy | Pos, Med | Amusement | Joy |
| Neutral / Descriptive (Wasfi) | وصفي | *Nabati-unique — no Ekman equivalent* | *Nabati-unique — no Plutchik equivalent* | Neutral, Low | *Nabati-unique* | *Nabati-unique* |

### 5.2 External Emotions → Nabati Manifestation (for emotions with no standalone Nabati class)

Emotions present in reference taxonomies that do not correspond to a standalone Nabati emotion class, and where they manifest in the tradition.

| External Emotion | System(s) | Nabati Manifestation | Form of Presence |
|-----------------|-----------|---------------------|-----------------|
| **Fear** (*khawf*, خوف) | Ekman, Plutchik, NRC | Subtext in Longing (Shawq), Sorrow (Huzn), Ritha; negated explicitly in Pride (Fakhr) and Defiance (Tahaddi) | Present but not performed as a primary aesthetic state. At low intensity: anxiety of separation in Longing and Ghazal; dread of permanent loss in Sorrow and Ritha; unease before fate in Hikma. The explicit negation *mani khayif* (ما أني خايف) in Fakhr/Defiance presupposes fear and publicly defeats it. |
| **Disgust** (*istinkar*, استنكار) | Ekman, Plutchik, NRC | Animating moral affect of Hija (Satire & Social Critique) genre | Encoded at genre level rather than as a private emotion class. Disgust is ritualised and outward-directed in Nabati oral tradition — the poet performs social moral disapproval as a public act with collective consequences. |
| **Surprise / Wonder** (*'ajab*, عجب) | Ekman, Plutchik | Sub-register within Admiration (I'jab) and Contemplation (Ta'ammul) | The root of I'jab is etymologically wonder (*'ajaba*); struck amazement at beauty or virtue is embedded in Admiration. In Contemplation, encountering the sublime in the desert landscape or the vastness of time produces an awed, wonder-adjacent state that blends with reflection. |
| **Anger** (*ghadhab*, غضب) | Ekman, Plutchik, NRC | Sublimated into Defiance (Tahaddi) and partially into Hija | Raw anger is not a celebrated performative aesthetic in Nabati tradition. It is channelled into controlled verbal assertion (Defiance) or social critique (Hija). The tradition's honour code valorises composed strength over uncontrolled rage. |

---

## References

1. **Holes, C. and Abu Athera, S. S.** *The Nabati Poetry of the United Arab Emirates.* Ithaca Press, Reading, 2011. ISBN 978-0-86372-383-4. — Primary cultural authority for Nabati genre definitions and thematic structure.

2. **Haider, M., Kim, E., Kuhn, G., and Zweig, G.** "PO-EMO: Conceptualization, Annotation, and Modeling of Aesthetic Emotions in German and English Poetry." *Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020)*, pp. 1652–1663. [ACL Anthology](https://aclanthology.org/2020.lrec-1.205/) — Closest reference taxonomy for poetry-specific aesthetic emotion.

3. **Ekman, P.** "An argument for basic emotions." *Cognition and Emotion* 6(3–4), 169–200, 1992. — Six universal basic emotions; baseline cross-reference.

4. **Plutchik, R.** *Emotion: A Psychoevolutionary Synthesis.* Harper & Row, New York, 1980. — Eight-emotion wheel; cross-reference.

5. **Russell, J. A.** "A circumplex model of affect." *Journal of Personality and Social Psychology* 39(6), 1161–1178, 1980. — Two-dimensional valence-arousal space.

6. **Mohammad, S. M. and Turney, P. D.** "Crowdsourcing a Word-Emotion Association Lexicon." *Computational Intelligence* 29(3), 436–465, 2013. — NRC Emotion Lexicon.

7. **Al-Jurjani, Abd al-Qadir.** *Asrar al-Balagha* (Secrets of Rhetoric). Dar al-Ma'arif, Cairo, 1959. — Classical Arabic rhetorical tradition; basis for *aghrad* genre comparison.

8. **Qudama ibn Ja'far.** *Naqd al-Shi'r* (Critique of Poetry). 9th century CE. — Foundational classical Arabic poetics; primary source for classical genre taxonomy.

9. **Qarah, F.** "AraPoemBERT: A Pretrained Language Model for Arabic Poetry Analysis." *ArabicNLP 2022.* — Backbone model for Nabati text analysis; trained on Arabic poetry corpora.

10. **Szreder, M. and Derrick, D.** "Phonological conditioning of affricate variability in Emirati Arabic." *Journal of the International Phonetic Association*, 2024. — Gulf Arabic phonology reference.

11. **Al Abdan.** الظواهر الصوتية في اللهجة الكويتية. جامعة آل البيت، 2018. — Gulf Arabic phonological phenomena.

---

*This document presents the full, unmerged taxonomies as a research reference. For project-specific label encoding and class merging decisions adopted in the Nabat-AI system, see `src/data/labels.py` and `docs/final_report.md`.*
