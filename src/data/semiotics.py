"""Semiotic lexicon for Nabati / Khaleeji poetry imagery tags.

Each entry maps a corpus imagery tag (lowercase) to its semiotic reading
in Gulf poetic tradition — covering the signified concept, cultural
connotation in Nabati verse, Peircean sign category, and the Arabic keyword.

Usage:
    from data.semiotics import lookup_semiotics, SEMIOTIC_LEXICON

    entry = lookup_semiotics("heart")
    # {'category': 'Body & Soul', 'arabic': 'قلب', 'signified': '...', 'connotation': '...'}
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

SEMIOTIC_LEXICON: dict[str, dict[str, str]] = {
    # ── Body & Soul ──────────────────────────────────────────────────────────
    "heart": {
        "category": "Body & Soul",
        "arabic": "قلب",
        "signified": "Seat of longing, devotion, and vulnerability",
        "connotation": (
            "In Nabati poetry the heart (qalb) is at once the wound and the witness — "
            "it registers love, betrayal, and grief as visceral, physical events rather "
            "than abstract states. A poet who 'gives the heart' surrenders entirely."
        ),
    },
    "eyes": {
        "category": "Body & Soul",
        "arabic": "عيون",
        "signified": "The gaze — of the beloved, the poet, or fate",
        "connotation": (
            "Eyes (ʿuyūn) are the primary site of beauty and of wound: the beloved's "
            "eyes strike like arrows; the poet's eyes weep as public testimony of "
            "sincerity. Sleepless eyes are proof that the depth of longing is real."
        ),
    },
    "tears": {
        "category": "Body & Soul",
        "arabic": "دموع",
        "signified": "Public proof of inner grief",
        "connotation": (
            "Tears (dumūʿ) are not merely emotion but evidence — displayed before an "
            "audience as proof that the poet's pain is genuine. Weeping legitimises "
            "the poem and invites communal empathy and solidarity."
        ),
    },
    "blood": {
        "category": "Body & Soul",
        "arabic": "دم",
        "signified": "Sacrifice, lineage, and the tribal covenant",
        "connotation": (
            "Blood (dam) evokes tribal honor, the price of betrayal, and genealogical "
            "bond. To spill blood is to enact the ultimate statement of commitment or "
            "vengeance; it binds generations and legitimises ancestral claims."
        ),
    },
    "hands": {
        "category": "Body & Soul",
        "arabic": "يدان",
        "signified": "Agency, generosity, and the giving or withholding of gifts",
        "connotation": (
            "Hands (yadān) perform the most socially visible acts: pouring coffee, "
            "raising the sword, offering alms. Open hands signal magnanimity; "
            "withheld hands mark the miserly patron who is shamed in verse."
        ),
    },
    # ── Cosmos ───────────────────────────────────────────────────────────────
    "moon": {
        "category": "Cosmos",
        "arabic": "قمر",
        "signified": "The beloved's face; hope and guidance at night",
        "connotation": (
            "The moon (qamar) is the Nabati poet's most versatile symbol: it illuminates "
            "the beloved's beauty, guides the night traveller, and marks time's passage. "
            "A full moon is the zenith of desire — radiant, unreachable, and brief."
        ),
    },
    "sun": {
        "category": "Cosmos",
        "arabic": "شمس",
        "signified": "Sovereign power, clarity, and the unforgiving heat of reality",
        "connotation": (
            "Unlike the gentle moon, the sun (shams) signifies sovereign power and "
            "the unforgiving heat of the Arabian day. It can represent the ruler, "
            "absolute truth, or the burning intensity of a separation that cannot be hidden."
        ),
    },
    "stars": {
        "category": "Cosmos",
        "arabic": "نجوم",
        "signified": "Navigation, fate, and the beloved seen from afar",
        "connotation": (
            "Stars (nujūm) guide Bedouin travellers across trackless desert and are "
            "read as fate's script. A poet who compares the beloved to a star places "
            "them beyond reach — beautiful, constant, but fundamentally untouchable."
        ),
    },
    "night": {
        "category": "Cosmos",
        "arabic": "ليل",
        "signified": "Reflection, secrecy, insomnia, and enforced separation",
        "connotation": (
            "Night (layl) is the poet's confidant: it conceals longing from society "
            "while allowing the heart to speak freely. Insomnia is its signature motif — "
            "the inability to sleep is presented as the surest proof of devotion."
        ),
    },
    "dawn": {
        "category": "Cosmos",
        "arabic": "فجر",
        "signified": "Hope, the end of the long night, and spiritual renewal",
        "connotation": (
            "Dawn (fajr) ends the night of longing and reflection, signalling the "
            "beginning of action and the answer to supplication. The call to prayer "
            "at dawn binds the cosmic and the devotional into one single moment."
        ),
    },
    # ── Landscape ────────────────────────────────────────────────────────────
    "desert": {
        "category": "Landscape",
        "arabic": "صحراء",
        "signified": "The Arabian heartland — solitude, freedom, and ordeal",
        "connotation": (
            "The desert (ṣaḥrāʾ) is both prison and paradise: a space of spiritual "
            "purification, tribal identity, and existential solitude. The ability to "
            "endure its silence and heat signals noble, unbreakable character."
        ),
    },
    "sand": {
        "category": "Landscape",
        "arabic": "رمال",
        "signified": "Impermanence, the vastness of time, and tribal homeland",
        "connotation": (
            "Sand (rimāl) records and erases: footprints vanish, names are forgotten. "
            "Yet the dunes of Najd or the Empty Quarter ground the poet's identity — "
            "home encoded in terrain, memory held in the shape of a dune."
        ),
    },
    "sea": {
        "category": "Landscape",
        "arabic": "بحر",
        "signified": "Vastness, the unknown, and the pearl-diving heritage",
        "connotation": (
            "For Gulf poets the sea (baḥr) carries the memory of pearl diving — "
            "the perilous breath-hold dive for beauty at mortal risk. It also signifies "
            "the unfathomable depth of grief or love that cannot be measured."
        ),
    },
    "palm tree": {
        "category": "Landscape",
        "arabic": "نخلة",
        "signified": "Steadfastness, provision, and the anchored homeland",
        "connotation": (
            "The palm tree (nakhla) feeds, shelters, and endures — it is the opposite "
            "of the wandering Bedouin, the fixed point that marks home. To praise "
            "someone as a palm is to praise their generosity and rootedness."
        ),
    },
    # ── Nature ───────────────────────────────────────────────────────────────
    "wind": {
        "category": "Nature",
        "arabic": "ريح",
        "signified": "The messenger; change, transience, and longing carried across distance",
        "connotation": (
            "Wind (rīḥ) carries the scent of the beloved from a distant land and "
            "delivers unspoken messages between separated hearts. It also signals "
            "impermanence — what the wind brings to you, it can also take away."
        ),
    },
    "fire": {
        "category": "Nature",
        "arabic": "نار",
        "signified": "Consuming passion, purification, and mortal danger",
        "connotation": (
            "Fire (nār) is the archetypal sign of consuming love: the poet 'burns' "
            "for the beloved in a flame that cannot be extinguished. It also signals "
            "the campfire of tribal hospitality and the pyre of collective grief."
        ),
    },
    "water": {
        "category": "Nature",
        "arabic": "ماء",
        "signified": "Life itself; purification and desperate longing",
        "connotation": (
            "In an arid landscape, water (māʾ) is life itself. The poet who is denied "
            "the beloved is like a desert with no rain — the image fuses ecological "
            "and emotional drought into a single felt reality."
        ),
    },
    "rain": {
        "category": "Nature",
        "arabic": "مطر",
        "signified": "Divine generosity, renewal, and tears from the sky",
        "connotation": (
            "Rain (maṭar) is among the most celebrated blessings in Gulf poetry — a "
            "gift from God that greens the land and gives life to the tribe. A poem of "
            "praise often likens the patron's generosity to a life-giving downpour."
        ),
    },
    "cloud": {
        "category": "Nature",
        "arabic": "غيوم",
        "signified": "Promise deferred and the veil over clarity",
        "connotation": (
            "Clouds (ghuyūm) promise rain but may pass without delivering — a sign of "
            "hope deferred or a patron's undecided mercy. Dark clouds before a storm "
            "signal impending change; their passing without rain is bitter irony."
        ),
    },
    # ── Creatures ────────────────────────────────────────────────────────────
    "falcon": {
        "category": "Creatures",
        "arabic": "صقر",
        "signified": "Sovereignty, nobility, and the disciplined hunting instinct",
        "connotation": (
            "The falcon (ṣaqr) is the supreme emblem of Gulf identity: trained, "
            "disciplined, fierce, and loyal. It represents the ruler's authority, "
            "the poet's proud spirit, and the relentless chase of an elusive quarry."
        ),
    },
    "horse": {
        "category": "Creatures",
        "arabic": "حصان",
        "signified": "Tribal nobility, freedom, and martial valor",
        "connotation": (
            "The Arabian horse (ḥiṣān) is nobility made flesh: swift, loyal, and "
            "beautiful. A poet praised as 'like a horse' is being likened to the "
            "finest quality the tribe can produce — pedigree embodied in speed."
        ),
    },
    "camel": {
        "category": "Creatures",
        "arabic": "جمل",
        "signified": "Endurance, the long journey, and Bedouin self-reliance",
        "connotation": (
            "The camel (jamal) embodies the virtues of desert life: patience under "
            "load, survival without water, unwavering loyalty. A poem about a camel "
            "is often a poem about perseverance — the will to endure without complaint."
        ),
    },
    "dove": {
        "category": "Creatures",
        "arabic": "حمامة",
        "signified": "Longing, mourning, and the voice that speaks for separation",
        "connotation": (
            "The dove (ḥamāma) calls out in the poet's place — its cooing at dawn "
            "mirrors the ache of the separated lover. It is the bird of grief, "
            "giving lament a natural voice before the poet has found their own words."
        ),
    },
    # ── Objects ──────────────────────────────────────────────────────────────
    "sword": {
        "category": "Objects",
        "arabic": "سيف",
        "signified": "Honor, justice, and the arbitration of tribal conflict",
        "connotation": (
            "The sword (sayf) is the poet-warrior's ultimate symbol: it enacts "
            "tribal justice, defends honor, and resolves disputes that words cannot. "
            "A tongue 'sharper than a sword' is the most dangerous of all weapons."
        ),
    },
    "coffee": {
        "category": "Objects",
        "arabic": "قهوة",
        "signified": "Hospitality, ceremony, and the ritual of tribal belonging",
        "connotation": (
            "Bitter coffee (gahwa) poured from a dallah is among the highest acts "
            "of Khaleeji hospitality. To offer coffee is to offer kinship; to refuse "
            "it is a serious affront. Its scent alone is enough to evoke home."
        ),
    },
    "dallah": {
        "category": "Objects",
        "arabic": "دلة",
        "signified": "The vessel of hospitality and the symbol of Gulf tradition",
        "connotation": (
            "The dallah — the long-spouted coffee pot — is among the most recognisable "
            "symbols of Gulf culture, appearing on currency and monuments. Pouring from "
            "it signals generosity, ceremony, and the continuity of ancestral custom."
        ),
    },
    "pearl": {
        "category": "Objects",
        "arabic": "لؤلؤة",
        "signified": "Rare beauty, the reward of mortal risk, and Gulf heritage",
        "connotation": (
            "Pearls (luʾluʾ) were the Gulf's great treasure before oil — won at mortal "
            "cost by divers who plunged without oxygen into dark water. A beloved "
            "compared to a pearl is beautiful precisely because they cost everything."
        ),
    },
    "tent": {
        "category": "Objects",
        "arabic": "خيمة",
        "signified": "Home, hospitality, and the nomadic covenant with guests",
        "connotation": (
            "The tent (khayma) is home that can be folded and carried — freedom and "
            "rootedness in one object. The guest who enters the tent falls under the "
            "full protection of the host's honor; the law of shelter is absolute."
        ),
    },
    # ── Motion & Time ────────────────────────────────────────────────────────
    "journey": {
        "category": "Motion & Time",
        "arabic": "رحلة",
        "signified": "Life's passage, exile, and the quest for the beloved",
        "connotation": (
            "Journey (riḥla) is both literal — the caravan crossing the desert — and "
            "metaphorical: the soul's search for meaning, reunion, or mercy. Exile "
            "from the beloved transforms the journey into an act of prolonged suffering."
        ),
    },
    "time": {
        "category": "Motion & Time",
        "arabic": "زمن",
        "signified": "Mortality, nostalgia, and the impermanence of glory",
        "connotation": (
            "Time (zaman) is the great eraser: it takes youth, beauty, and the beloved. "
            "Nabati poets reckon with time as a rival — something to outwit through "
            "verse that will be sung long after the poet is gone and forgotten."
        ),
    },
    "exile": {
        "category": "Motion & Time",
        "arabic": "غربة",
        "signified": "Displacement, longing for homeland, and belonging denied",
        "connotation": (
            "Exile (ghurba) is spiritual homesickness: the poet far from tribe, beloved, "
            "or ancestral land. It generates the most keenly felt poetry in the Nabati "
            "tradition — sharpened by the specific, named landscape left behind."
        ),
    },
    # ── Emotions as Signs ────────────────────────────────────────────────────
    "longing": {
        "category": "Emotion as Sign",
        "arabic": "شوق",
        "signified": "The ache of absence; desire made permanent by distance",
        "connotation": (
            "Longing (shawq) is the engine of Nabati love poetry: the beloved is "
            "always elsewhere, making desire permanent rather than ever satisfied. "
            "The poem itself is the longing made audible — it exists because distance does."
        ),
    },
    "loss": {
        "category": "Emotion as Sign",
        "arabic": "فقد",
        "signified": "Grief, deprivation, and the void left by an irreversible absence",
        "connotation": (
            "Loss (faqd) covers both the death of a loved one and the loss of status, "
            "homeland, or the beloved's regard. It is among the most socially recognised "
            "of Nabati emotions — performed publicly at elegies and tribal mourning."
        ),
    },
    "hope": {
        "category": "Emotion as Sign",
        "arabic": "أمل",
        "signified": "Trust in the future; the breath that keeps the poem alive",
        "connotation": (
            "Hope (amal) is what sustains the poem: the possibility that the beloved "
            "will return, the ruler will hear the petition, or God will restore what "
            "was lost. Without hope there is only elegy — and elegy has no tomorrow."
        ),
    },
    "freedom": {
        "category": "Emotion as Sign",
        "arabic": "حرية",
        "signified": "Autonomy of spirit; the Bedouin's refusal to be owned",
        "connotation": (
            "Freedom (ḥurriyya) in Nabati poetry is inseparable from the desert: "
            "the open horizon that no ruler can fully claim. A poet who invokes freedom "
            "asserts the untameable dignity of the tribal self against all constraint."
        ),
    },
    "nostalgia": {
        "category": "Emotion as Sign",
        "arabic": "حنين",
        "signified": "Sweet grief for a past that cannot return",
        "connotation": (
            "Nostalgia (ḥanīn) in Gulf poetry is not passive sentimentality but an "
            "active, often painful summons of a vanished world — the tribal gathering, "
            "the ancestral home, the face of the beloved seen for the last time."
        ),
    },
}

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_ALIAS_MAP: dict[str, str] = {
    "qalb": "heart",
    "layla": "night",
    "falcon": "falcon",
    "gahwa": "coffee",
    "dallah": "dallah",
    "nakhla": "palm tree",
    "pearl diving": "pearl",
    "camel": "camel",
    "hawk": "falcon",
    "eagle": "falcon",
}


def lookup_semiotics(tag: str) -> dict[str, str] | None:
    """Return semiotic entry for *tag*, with alias + partial-match fallback.

    Returns ``None`` if no entry is found.
    """
    key = tag.strip().lower()
    # Direct lookup
    if key in SEMIOTIC_LEXICON:
        return SEMIOTIC_LEXICON[key]
    # Alias lookup
    if key in _ALIAS_MAP:
        mapped = _ALIAS_MAP[key]
        if mapped in SEMIOTIC_LEXICON:
            return SEMIOTIC_LEXICON[mapped]
    # Partial match — key is substring of lexicon key or vice versa
    for lex_key, entry in SEMIOTIC_LEXICON.items():
        if lex_key in key or key in lex_key:
            return entry
    return None
