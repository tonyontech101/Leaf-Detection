"""
Static, offline plant reference data + status-based care guidance.

This module enriches the analysis response with human-readable context:
  * scientific name + a short description for each dataset class
  * severity / symptoms / treatment / prevention text derived from the
    health status produced by ``health.analyze``.

Everything here is local, deterministic data - no models, no network. The
species classifier does not diagnose named diseases, so the "disease" field
is a status-derived summary rather than a specific pathology. This keeps the
UI honest while still presenting actionable, structured guidance.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Reference data for the 48 dataset classes.
# scientific_name is left blank ("") for local names that are not reliably
# mapped to a single species; the UI renders a graceful placeholder.
# --------------------------------------------------------------------------
PLANTS: dict[str, dict[str, str]] = {
    "Aloevera":        {"scientific_name": "Aloe barbadensis miller", "description": "A succulent whose thick, gel-filled leaves are widely used in skin care and traditional remedies."},
    "Amla":            {"scientific_name": "Phyllanthus emblica", "description": "Indian gooseberry; a deciduous tree valued for its vitamin-C-rich fruit and medicinal leaves."},
    "Amruthaballi":    {"scientific_name": "Tinospora cordifolia", "description": "A climbing shrub (Guduchi) with heart-shaped leaves, prominent in Ayurvedic medicine."},
    "Arali":           {"scientific_name": "Nerium oleander", "description": "An ornamental evergreen shrub with lance-shaped leaves; all parts are toxic if ingested."},
    "Astma_weed":      {"scientific_name": "Euphorbia hirta", "description": "A small herb traditionally used for respiratory complaints, with hairy stems and leaves."},
    "Badipala":        {"scientific_name": "", "description": "A regional medicinal plant included in the reference dataset."},
    "Balloon_Vine":    {"scientific_name": "Cardiospermum halicacabum", "description": "A climbing vine with balloon-like seed capsules and finely divided leaves."},
    "Bamboo":          {"scientific_name": "Bambusa vulgaris", "description": "A fast-growing woody grass with slender, blade-like leaves."},
    "Beans":           {"scientific_name": "Phaseolus vulgaris", "description": "The common bean; a legume with trifoliate leaves grown as a food crop."},
    "Betel":           {"scientific_name": "Piper betle", "description": "A vine grown for its aromatic, heart-shaped leaves used in traditional chewing preparations."},
    "Bhrami":          {"scientific_name": "Bacopa monnieri", "description": "A creeping marsh herb (Brahmi) with small succulent leaves, used as a nootropic in Ayurveda."},
    "Caricature":      {"scientific_name": "Graptophyllum pictum", "description": "An ornamental shrub known for its patterned, variegated foliage."},
    "Castor":          {"scientific_name": "Ricinus communis", "description": "A vigorous plant with large palmate leaves; source of castor oil."},
    "Catharanthus":    {"scientific_name": "Catharanthus roseus", "description": "Madagascar periwinkle; a flowering plant with glossy leaves and medicinal alkaloids."},
    "Chakte":          {"scientific_name": "", "description": "A regional medicinal plant included in the reference dataset."},
    "Chilly":          {"scientific_name": "Capsicum annuum", "description": "The chilli pepper; a cultivated plant with simple ovate leaves."},
    "Citron lime (herelikai)": {"scientific_name": "Citrus medica", "description": "A citrus shrub with fragrant leaves and thick-rind fruit."},
    "Common rue(naagdalli)":   {"scientific_name": "Ruta graveolens", "description": "An aromatic evergreen herb (rue) with bluish, deeply lobed leaves."},
    "Coriender":       {"scientific_name": "Coriandrum sativum", "description": "Coriander/cilantro; a soft-leaved annual herb used in cooking."},
    "Curry":           {"scientific_name": "Murraya koenigii", "description": "The curry leaf tree, prized for its aromatic pinnate leaves."},
    "Doddpathre":      {"scientific_name": "Plectranthus amboinicus", "description": "Indian borage; a fleshy, aromatic herb used for coughs and colds."},
    "Drumstick":       {"scientific_name": "Moringa oleifera", "description": "A fast-growing tree with nutritious leaves and edible seed pods."},
    "Ekka":            {"scientific_name": "Calotropis gigantea", "description": "A hardy shrub (crown flower) with large, pale, felted leaves."},
    "Eucalyptus":      {"scientific_name": "Eucalyptus globulus", "description": "An evergreen tree with aromatic, oil-rich leaves."},
    "Gasagase":        {"scientific_name": "Papaver somniferum", "description": "The poppy plant, with lobed bluish-green leaves; source of poppy seeds."},
    "Ginger":          {"scientific_name": "Zingiber officinale", "description": "A rhizomatous herb with long lance-shaped leaves; the rhizome is a common spice."},
    "Globe Amarnath":  {"scientific_name": "Gomphrena globosa", "description": "Globe amaranth; an ornamental with clover-like flower heads and simple leaves."},
    "Guava":           {"scientific_name": "Psidium guajava", "description": "A tropical fruit tree with leathery, veined leaves used in herbal teas."},
    "Henna":           {"scientific_name": "Lawsonia inermis", "description": "A shrub whose leaves yield the natural reddish dye henna."},
    "Hibiscus":        {"scientific_name": "Hibiscus rosa-sinensis", "description": "An ornamental shrub with glossy leaves and large showy flowers."},
    "Honge":           {"scientific_name": "Millettia pinnata", "description": "Indian beech (Pongamia); a tree with glossy compound leaves, valued for biofuel oil."},
    "Insulin":         {"scientific_name": "Costus igneus", "description": "The insulin plant, with spiralling fleshy leaves used in folk diabetes remedies."},
    "Jackfruit":       {"scientific_name": "Artocarpus heterophyllus", "description": "A large tropical tree with thick, glossy leaves and the world's largest tree-borne fruit."},
    "Jasmine":         {"scientific_name": "Jasminum officinale", "description": "A climbing shrub with fragrant white flowers and pinnate leaves."},
    "Kambajala":       {"scientific_name": "", "description": "A regional medicinal plant included in the reference dataset."},
    "Kasambruga":      {"scientific_name": "", "description": "A regional medicinal plant included in the reference dataset."},
    "Lemon":           {"scientific_name": "Citrus limon", "description": "A citrus tree with aromatic leaves and acidic yellow fruit."},
    "Malabar_Spinach": {"scientific_name": "Basella alba", "description": "A leafy climbing vine grown as a spinach substitute in warm climates."},
    "Mango":           {"scientific_name": "Mangifera indica", "description": "A tropical tree with lance-shaped leaves, cultivated for its sweet fruit."},
    "Marigold":        {"scientific_name": "Tagetes erecta", "description": "An ornamental annual with pinnate leaves and bright pom-pom flowers."},
    "Mint":            {"scientific_name": "Mentha", "description": "An aromatic herb with serrated leaves used widely in cooking and teas."},
    "Neem":            {"scientific_name": "Azadirachta indica", "description": "A hardy tree with pinnate leaves, renowned for its medicinal and pesticidal properties."},
    "Nelavembu":       {"scientific_name": "Andrographis paniculata", "description": "King of bitters; a herb with lance-shaped leaves used against fevers."},
    "Rose":            {"scientific_name": "Rosa indica", "description": "A woody shrub with pinnate, serrated leaflets and fragrant flowers."},
    "Seethaashoka":    {"scientific_name": "Saraca asoca", "description": "The Ashoka tree, with drooping compound leaves, sacred in Indian tradition."},
    "Tomato":          {"scientific_name": "Solanum lycopersicum", "description": "A cultivated plant with compound leaves grown for its edible fruit."},
    "Turmeric":        {"scientific_name": "Curcuma longa", "description": "A rhizomatous herb with broad leaves; the rhizome yields the spice turmeric."},
    "ashoka":          {"scientific_name": "Polyalthia longifolia", "description": "The false ashoka (mast tree), with wavy-edged drooping evergreen leaves."},
}

_FALLBACK = {"scientific_name": "", "description": "A plant species from the 48-class reference dataset."}


# --------------------------------------------------------------------------
# Status-derived care guidance. The classifier is not a pathology model, so
# these are general, honest recommendations keyed off the visual status.
# --------------------------------------------------------------------------
_CARE: dict[str, dict[str, str]] = {
    "HEALTHY": {
        "disease": "None detected",
        "severity": "None",
        "symptoms": "No visible discoloration, spotting, or wilting was detected on the leaf surface.",
        "treatment": "No treatment required. Continue routine care and monitor periodically.",
        "prevention": "Maintain consistent watering, adequate sunlight, and good airflow. Inspect new growth regularly for early signs of stress.",
    },
    "MINOR_ISSUES": {
        "disease": "Minor stress or early symptoms",
        "severity": "Low to moderate",
        "symptoms": "Some discoloration, light spotting, or edge browning may be present. Damage appears limited and early-stage.",
        "treatment": "Remove affected leaves, avoid overhead watering, and improve airflow. Consider a mild organic fungicide or neem-oil spray if symptoms spread.",
        "prevention": "Water at the base rather than the foliage, avoid overcrowding, and keep tools clean to limit spread.",
    },
    "UNHEALTHY": {
        "disease": "Significant stress or disease indicators",
        "severity": "High",
        "symptoms": "Noticeable yellowing, browning, spotting, holes, or wilting suggests active disease or environmental stress.",
        "treatment": "Isolate the plant, prune and dispose of heavily affected foliage, and apply an appropriate treatment. Consult a horticultural expert for a targeted diagnosis.",
        "prevention": "Quarantine new plants, sanitise tools, ensure proper drainage, and correct light or nutrient imbalances.",
    },
    "UNKNOWN": {
        "disease": "Not determined",
        "severity": "Unknown",
        "symptoms": "The leaf condition could not be assessed confidently from the image.",
        "treatment": "Retake the photo with a single, well-lit leaf on a plain background for a clearer assessment.",
        "prevention": "Continue routine care and re-scan if you notice changes in colour or texture.",
    },
}


def describe(species_name: str) -> dict[str, str]:
    """Return scientific name + description for a dataset class name."""
    return PLANTS.get(species_name, _FALLBACK)


def care(status: str, observations: str = "") -> dict[str, str]:
    """
    Build the care/guidance block for a given health status.

    If a local VLM produced free-text ``observations`` they are surfaced as
    the primary symptom description; otherwise the status default is used.
    """
    base = _CARE.get(status, _CARE["UNKNOWN"])
    out = dict(base)
    obs = (observations or "").strip()
    if obs:
        out["symptoms"] = obs
    return out
